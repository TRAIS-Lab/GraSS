from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Callable, Any, Optional, Tuple, List, Union
if TYPE_CHECKING:
    from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import functools
import time


class HookManager:
    """
    Manages hooks for efficient gradient component capturing and projection
    without requiring custom layer implementations.

    Supports two modes of gradient projection:
    1. Component-wise: Projects input and grad_pre_activation separately before reconstruction
    2. Direct gradient: Uses PyTorch's parameter.grad directly and applies projection
    """
    def __init__(
            self,
            model: nn.Module,
            layer_names: List[str],
            profile: bool = False,
            device: str = 'cpu'
        ) -> None:
        """
        Initialize the hook manager

        Args:
            model: The model to hook
            layer_names: Names of layers to hook
            profile: Whether to profile execution time
            device: Device to use for profiling synchronization
        """
        self.model = model
        self.layer_names = layer_names
        self.profile = profile
        self.device = device

        # Create mapping from layer name to index for O(1) lookups
        self.layer_name_to_idx = {name: idx for idx, name in enumerate(layer_names)}

        # Maps layer names to modules for direct gradient access
        self.layer_modules = {name: None for name in layer_names}

        self.forward_hooks = [None] * len(layer_names)
        self.backward_hooks = [None] * len(layer_names)
        self.param_grad_hooks = [None] * len(layer_names)  # New: hooks for direct parameter gradients
        self.projected_grads = [None] * len(layer_names)
        self.inputs = [None] * len(layer_names)
        self.pre_activations = [None] * len(layer_names)
        self.normalized = [None] * len(layer_names)  # For LayerNorm
        self.projectors = [None] * len(layer_names)

        # Flag to determine which mode to use for each layer
        self.use_direct_grad = [False] * len(layer_names)

        # Profiling stats
        self.projection_time = 0.0

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks to target layers"""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                idx = self.layer_name_to_idx[name]
                self.layer_modules[name] = module

                # Use functools.partial to correctly bind parameters to avoid late binding issues
                forward_hook = functools.partial(self._forward_hook_fn, name)
                backward_hook = functools.partial(self._backward_hook_fn, name)

                # Register hooks with properly bound parameters
                self.forward_hooks[idx] = module.register_forward_hook(forward_hook)
                self.backward_hooks[idx] = module.register_full_backward_hook(backward_hook)

                # We'll register param_grad_hooks dynamically later when needed

    def set_projectors(self, projectors: List[Any]) -> None:
        """
        Set projector objects for each layer and determine which hook mode to use.

        If projector has only 'projector_grad' and not 'projector_grad_comp',
        we'll use direct gradient mode.

        Args:
            projectors: List of projector objects, ordered by layer_names
        """
        self.projectors = projectors

        # Determine which hook mode to use for each layer
        for idx, projector in enumerate(projectors):
            if projector is None:
                continue

            # Check if we should use direct gradient mode
            if (hasattr(projector, 'projector_grad') and projector.projector_grad is not None and
                (not hasattr(projector, 'projector_grad_comp') or
                 projector.projector_grad_comp == (None, None))):

                self.use_direct_grad[idx] = True
                # Register parameter gradient hooks for this layer
                self._register_param_grad_hooks(idx)
            else:
                self.use_direct_grad[idx] = False

    def _register_param_grad_hooks(self, idx: int):
        """
        Register hooks to capture and project parameter gradients directly

        Args:
            idx: Layer index to register hooks for
        """
        layer_name = self.layer_names[idx]
        module = self.layer_modules[layer_name]

        # Remove existing backward hook as we'll use param grad hooks instead
        if self.backward_hooks[idx] is not None:
            self.backward_hooks[idx].remove()
            self.backward_hooks[idx] = None

        # Create hooks for each parameter in the module
        hooks = []

        def make_hook_fn(param_name, idx=idx):
            def hook_fn(grad):
                # Apply projection to the gradient
                self._project_param_grad(idx, param_name, grad)
                return grad
            return hook_fn

        # Register hooks for each parameter
        for name, param in module.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(make_hook_fn(name))
                hooks.append(hook)

        self.param_grad_hooks[idx] = hooks

    def _project_param_grad(self, idx: int, param_name: str, grad: Tensor):
        """
        Project parameter gradients directly using projector_grad

        Args:
            idx: Layer index
            param_name: Parameter name
            grad: Gradient tensor
        """
        projector = self.projectors[idx]
        if projector and hasattr(projector, 'projector_grad') and projector.projector_grad is not None:
            # Start timing for projection if profiling is enabled
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                start_time = time.time()

            # For direct gradient mode, we treat the param grad as a batch of 1
            # We need to reshape and then reshape back
            orig_shape = grad.shape

            # Reshape to match projector's expected input
            batch_grad = grad.reshape(1, -1)

            # Apply projection
            projected_grad = projector.projector_grad(batch_grad)

            # Reshape back to original shape
            if projected_grad.numel() == grad.numel():  # Check if dimensions match
                projected_grad = projected_grad.reshape(orig_shape)

                # Store a copy for later retrieval
                if self.projected_grads[idx] is None:
                    self.projected_grads[idx] = {}

                self.projected_grads[idx][param_name] = projected_grad.detach().clone()

                # In-place modification of the gradient
                grad.copy_(projected_grad)

            # End timing for projection
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                self.projection_time += time.time() - start_time

    def get_projected_grads(self) -> List[Union[Tensor, Dict[str, Tensor]]]:
        """
        Get all captured projected gradients.
        For direct gradient mode, returns a dict mapping parameter names to gradients.

        Returns:
            List of projected gradient tensors or dictionaries, ordered by layer_names
        """
        return self.projected_grads

    def get_projection_time(self) -> float:
        """
        Get the accumulated projection time

        Returns:
            Total time spent in projection operations
        """
        return self.projection_time

    def _forward_hook_fn(self, name: str, mod: nn.Module, inp: Any, out: Any) -> None:
        """
        Forward hook function that captures inputs and pre-activations

        Args:
            name: Layer name
            mod: Module instance
            inp: Input tensors
            out: Output tensors
        """
        # Get the index for this layer
        idx = self.layer_name_to_idx[name]

        # Skip if we're using direct gradient mode (we still need forward hooks for capturing inputs)
        # This data is still useful for debugging or analysis

        # Store input
        if isinstance(inp, tuple) and len(inp) > 0:
            self.inputs[idx] = inp[0].detach()
        else:
            self.inputs[idx] = inp.detach()

        # Store pre-activation (output)
        self.pre_activations[idx] = out.detach()

        # For LayerNorm, also capture the normalized tensor
        if isinstance(mod, nn.LayerNorm):
            x = inp[0] if isinstance(inp, tuple) else inp
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            normalized = (x - mean) / torch.sqrt(var + mod.eps)
            self.normalized[idx] = normalized.detach()

    def _backward_hook_fn(self, name: str, mod: nn.Module, grad_input: Any, grad_output: Any) -> None:
        """
        Backward hook function that computes projected gradients
        Only used for component-wise projection mode

        Args:
            name: Layer name
            mod: Module instance
            grad_input: Gradient w.r.t inputs
            grad_output: Gradient w.r.t outputs
        """
        # Get the index for this layer
        idx = self.layer_name_to_idx[name]

        # Skip if we're using direct gradient mode
        if self.use_direct_grad[idx]:
            return

        # Get the gradient of the pre-activation
        grad_pre_activation = grad_output[0]

        # Calculate the projected gradient based on layer type
        with torch.no_grad():
            if isinstance(mod, nn.Linear):
                grad = self._linear_grad_from_grad_comp(
                    mod, idx, grad_pre_activation, per_sample=True
                )
            elif isinstance(mod, nn.LayerNorm):
                grad = self._layernorm_grad_from_grad_comp(
                    mod, idx, grad_pre_activation, per_sample=True
                )
            elif isinstance(mod, nn.Embedding):
                # Embeddings would need their own implementation
                grad = None
            else:
                # Fallback for other layer types
                grad = None

            if grad is not None:
                # Store the projected gradient
                self.projected_grads[idx] = grad.detach()

    def _linear_grad_from_grad_comp(
        self,
        layer: nn.Linear,
        idx: int,
        grad_pre_activation: Tensor,
        per_sample: bool = True
    ) -> Tensor:
        """
        Compute the gradient for Linear layers using component-wise projection

        Args:
            layer: Linear layer
            idx: Layer index
            grad_pre_activation: Gradient of the pre-activation
            per_sample: Whether to compute per-sample gradients

        Returns:
            Projected gradient tensor
        """
        input_features = self.inputs[idx]
        is_3d = input_features.dim() == 3

        if is_3d:
            batch_size, seq_length, hidden_size = input_features.shape
            # Reshape 3D tensors to 2D for consistent processing
            input_features = input_features.reshape(-1, hidden_size)
            grad_pre_activation = grad_pre_activation.reshape(-1, layer.out_features)
        else:
            batch_size = input_features.shape[0]

        # Scale the gradient if we're computing per-sample gradients
        if per_sample:
            grad_pre_activation = grad_pre_activation * batch_size

        # Handle bias term by augmenting input with ones
        if layer.bias is not None:
            ones = torch.ones(
                input_features.size(0), 1,
                device=input_features.device,
                dtype=input_features.dtype
            )
            input_features = torch.cat([input_features, ones], dim=1)

        if is_3d:
            # Reshape back to 3D
            input_features = input_features.reshape(batch_size, seq_length, -1)
            grad_pre_activation = grad_pre_activation.reshape(batch_size, seq_length, -1)

        # Apply projectors if they exist - only measure this part for profiling
        projector = self.projectors[idx]
        if projector and hasattr(projector, 'projector_grad_comp') and projector.projector_grad_comp != (None, None):
            # Start timing for projection if profiling is enabled
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                start_time = time.time()

            projector_grad_comp_1, projector_grad_comp_2 = projector.projector_grad_comp

            # Apply projection to gradient components
            grad_pre_activation_flatten = grad_pre_activation.view(-1, grad_pre_activation.shape[-1])
            input_features_flatten = input_features.view(-1, input_features.shape[-1])

            if is_3d:
                grad_pre_activation = projector_grad_comp_1(grad_pre_activation_flatten).view(
                    grad_pre_activation.shape[0], grad_pre_activation.shape[1], -1
                )
                input_features = projector_grad_comp_2(input_features_flatten).view(
                    input_features.shape[0], input_features.shape[1], -1
                )
            else:
                grad_pre_activation = projector_grad_comp_1(grad_pre_activation_flatten)
                input_features = projector_grad_comp_2(input_features_flatten)

            # End timing for projection
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                self.projection_time += time.time() - start_time

        # Compute the outer product to get the gradient
        if is_3d:
            grad = torch.einsum('ijk,ijl->ikl', grad_pre_activation, input_features).reshape(batch_size, -1)
        else:
            grad = torch.einsum('bi,bj->bij', grad_pre_activation, input_features).reshape(batch_size, -1)

        # Apply final projector if available - also measure this as part of projection
        if projector and hasattr(projector, 'projector_grad') and projector.projector_grad is not None:
            # Start timing for projection if profiling is enabled
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                start_time = time.time()

            grad = projector.projector_grad(grad)

            # End timing for projection
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                self.projection_time += time.time() - start_time

        return grad

    def _layernorm_grad_from_grad_comp(
        self,
        layer: nn.LayerNorm,
        idx: int,
        grad_pre_activation: Tensor,
        per_sample: bool = True
    ) -> Tensor:
        """
        Compute the gradient for LayerNorm layers using component-wise projection

        Args:
            layer: LayerNorm layer
            idx: Layer index
            grad_pre_activation: Gradient of the pre-activation
            per_sample: Whether to compute per-sample gradients

        Returns:
            Projected gradient tensor
        """
        if not layer.elementwise_affine:
            return None

        normalized = self.normalized[idx]
        if normalized is None:
            return None

        is_3d = normalized.dim() == 3

        if per_sample:
            grad_pre_activation = grad_pre_activation * normalized.shape[0]
            if is_3d:
                grad_weight = torch.einsum("ijk,ijk->ik", grad_pre_activation, normalized)
                grad_bias = torch.sum(grad_pre_activation, dim=1)
            else:
                grad_weight = grad_pre_activation * normalized
                grad_bias = grad_pre_activation
        else:
            if is_3d:
                grad_weight = torch.sum(grad_pre_activation * normalized, dim=(0, 1))
                grad_bias = torch.sum(grad_pre_activation, dim=(0, 1))
            else:
                grad_weight = torch.sum(grad_pre_activation * normalized, dim=0)
                grad_bias = torch.sum(grad_pre_activation, dim=0)

        # Apply projectors if they exist - only measure this part for profiling
        projector = self.projectors[idx]
        if projector and hasattr(projector, 'projector_grad_comp') and projector.projector_grad_comp != (None, None):
            # Start timing for projection if profiling is enabled
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                start_time = time.time()

            projector_grad_comp_1, projector_grad_comp_2 = projector.projector_grad_comp
            grad_weight = projector_grad_comp_1(grad_weight)
            grad_bias = projector_grad_comp_2(grad_bias)

            # End timing for projection
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                self.projection_time += time.time() - start_time

        # Concatenate weight and bias gradients
        grad = torch.cat((grad_weight, grad_bias), dim=1)

        # Apply final projector if available - also measure this as part of projection
        if projector and hasattr(projector, 'projector_grad') and projector.projector_grad is not None:
            # Start timing for projection if profiling is enabled
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                start_time = time.time()

            grad = projector.projector_grad(grad)

            # End timing for projection
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                self.projection_time += time.time() - start_time

        return grad

    def remove_hooks(self) -> None:
        """Remove all hooks"""
        for hook in self.forward_hooks:
            if hook is not None:
                hook.remove()
        for hook in self.backward_hooks:
            if hook is not None:
                hook.remove()

        # Remove parameter gradient hooks
        for hooks in self.param_grad_hooks:
            if hooks is not None:
                for hook in hooks:
                    hook.remove()

        self.forward_hooks = [None] * len(self.layer_names)
        self.backward_hooks = [None] * len(self.layer_names)
        self.param_grad_hooks = [None] * len(self.layer_names)