from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Callable, Any, Optional, Tuple, List
if TYPE_CHECKING:
    from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import functools
import time

# JIT-compiled helper functions for critical operations
@torch.jit.script
def compute_weight_gradients_3d(grad_output: torch.Tensor, input_features: torch.Tensor) -> torch.Tensor:
    """JIT-compiled function for efficient 3D weight gradient computation"""
    return torch.einsum('bso,bsi->boi', grad_output, input_features)

@torch.jit.script
def compute_weight_gradients_2d(grad_output: torch.Tensor, input_features: torch.Tensor) -> torch.Tensor:
    """JIT-compiled function for efficient 2D weight gradient computation"""
    return torch.einsum('bo,bi->boi', grad_output, input_features)


class HookManager:
    """
    Manages hooks for efficient gradient component capturing and projection
    without requiring custom layer implementations.
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

        self.forward_hooks = [None] * len(layer_names)
        self.backward_hooks = [None] * len(layer_names)
        self.projected_grads = [None] * len(layer_names)
        self.inputs = [None] * len(layer_names)
        self.pre_activations = [None] * len(layer_names)
        self.normalized = [None] * len(layer_names)  # For LayerNorm
        self.projectors = [None] * len(layer_names)

        # Profiling stats
        self.projection_time = 0.0

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks to target layers"""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                idx = self.layer_name_to_idx[name]

                # Use functools.partial to correctly bind parameters to avoid late binding issues
                forward_hook = functools.partial(self._forward_hook_fn, name)
                backward_hook = functools.partial(self._backward_hook_fn, name)

                # Register hooks with properly bound parameters
                self.forward_hooks[idx] = module.register_forward_hook(forward_hook)
                self.backward_hooks[idx] = module.register_full_backward_hook(backward_hook)

    def set_projectors(self, projectors: List[Any]) -> None:
        """
        Set projector objects for each layer

        Args:
            projectors: List of projector objects, ordered by layer_names
        """
        self.projectors = projectors

    def get_projected_grads(self) -> List[Tensor]:
        """
        Get all captured projected gradients

        Returns:
            List of projected gradient tensors, ordered by layer_names
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

        Args:
            name: Layer name
            mod: Module instance
            grad_input: Gradient w.r.t inputs
            grad_output: Gradient w.r.t outputs
        """
        # Get the index for this layer
        idx = self.layer_name_to_idx[name]

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
        Compute the gradient for Linear layers, with optimized path when only
        gradient projector is available.

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
        projector = self.projectors[idx]

        # Determine which projection approach to use
        using_component_projectors = (
            projector and
            hasattr(projector, 'projector_grad_comp') and
            projector.projector_grad_comp != (None, None)
        )
        using_grad_projector = (
            projector and
            hasattr(projector, 'projector_grad') and
            projector.projector_grad is not None
        )

        # Optimized path: When we only need to project the final gradient
        if not using_component_projectors and using_grad_projector:
            return self._compute_projected_param_gradients(layer, idx, input_features,
                                                          grad_pre_activation, per_sample)

        # Original path: When we need to project components
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

        # Apply component projectors if they exist
        if using_component_projectors:
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

        # Compute the outer product to get the gradient - use optimized einsum
        if is_3d:
            # Use JIT-compiled function for better performance
            grad_tensor = compute_weight_gradients_3d(grad_pre_activation, input_features)
            grad = grad_tensor.reshape(batch_size, -1)
        else:
            # Use JIT-compiled function for better performance
            grad_tensor = compute_weight_gradients_2d(grad_pre_activation, input_features)
            grad = grad_tensor.reshape(batch_size, -1)

        # Apply final projector if available
        if using_grad_projector:
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

    def _compute_projected_param_gradients(
        self,
        layer: nn.Linear,
        idx: int,
        input_features: Tensor,
        grad_pre_activation: Tensor,
        per_sample: bool = True
    ) -> Tensor:
        """
        Compute per-sample gradients for a linear layer using optimized operations.

        Args:
            layer: Linear layer
            idx: Layer index
            input_features: Input features to the layer
            grad_pre_activation: Gradient of the pre-activation
            per_sample: Whether to compute per-sample gradients

        Returns:
            Projected gradient tensor
        """
        projector = self.projectors[idx]
        batch_size = input_features.shape[0]

        with torch.no_grad():
            # Handle different input dimensions
            is_3d = input_features.dim() == 3

            if is_3d:
                # For 3D inputs (sequence data)
                batch_size, seq_length, hidden_size = input_features.shape
                out_features = layer.out_features

                if per_sample:
                    # Scale gradients for per-sample computation
                    grad_pre_activation = grad_pre_activation * batch_size

                    # Use JIT-compiled einsum for maximum efficiency
                    per_sample_grads_w = compute_weight_gradients_3d(
                        grad_pre_activation, input_features
                    )

                    # For bias gradients, sum over sequence dimension efficiently
                    if layer.bias is not None:
                        per_sample_grads_b = grad_pre_activation.sum(dim=1)  # B x O

                        # Combine weight and bias gradients efficiently
                        per_sample_grads = torch.cat([
                            per_sample_grads_w.reshape(batch_size, -1),
                            per_sample_grads_b
                        ], dim=1)
                    else:
                        per_sample_grads = per_sample_grads_w.reshape(batch_size, -1)
                else:
                    # For full batch, reshape once and use efficient matmul
                    input_reshaped = input_features.reshape(-1, hidden_size)
                    grad_reshaped = grad_pre_activation.reshape(-1, out_features)

                    # Use torch.mm which is more efficient for 2D matrices
                    per_sample_grads_w = torch.mm(grad_reshaped.t(), input_reshaped)

                    if layer.bias is not None:
                        per_sample_grads_b = grad_reshaped.sum(dim=0)
                        per_sample_grads = torch.cat([per_sample_grads_w.reshape(-1), per_sample_grads_b])
                    else:
                        per_sample_grads = per_sample_grads_w.reshape(-1)

                    # Efficient expansion using repeat instead of expand
                    per_sample_grads = per_sample_grads.repeat(batch_size, 1)
            else:
                # For 2D inputs (standard batch data)
                if per_sample:
                    # Scale gradients for per-sample computation
                    grad_pre_activation = grad_pre_activation * batch_size

                    # Use JIT-compiled einsum for outer product
                    per_sample_grads_w = compute_weight_gradients_2d(
                        grad_pre_activation, input_features
                    )

                    # For bias gradients
                    if layer.bias is not None:
                        per_sample_grads_b = grad_pre_activation  # B x O

                        # Reshape once and combine
                        per_sample_grads = torch.cat([
                            per_sample_grads_w.reshape(batch_size, -1),
                            per_sample_grads_b
                        ], dim=1)
                    else:
                        per_sample_grads = per_sample_grads_w.reshape(batch_size, -1)
                else:
                    # Use efficient matmul for full batch computation
                    per_sample_grads_w = torch.mm(grad_pre_activation.t(), input_features)

                    if layer.bias is not None:
                        per_sample_grads_b = grad_pre_activation.sum(dim=0)
                        per_sample_grads = torch.cat([per_sample_grads_w.reshape(-1), per_sample_grads_b])
                    else:
                        per_sample_grads = per_sample_grads_w.reshape(-1)

                    # Efficient expansion using repeat
                    per_sample_grads = per_sample_grads.repeat(batch_size, 1)

            # Apply gradient projector (ensure it runs on the same device)
            if projector and hasattr(projector, 'projector_grad') and projector.projector_grad is not None:
                # Start timing for the whole operation if profiling is enabled
                if self.profile:
                    torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                    start_time = time.time()
                per_sample_grads = projector.projector_grad(per_sample_grads)
                if self.profile:
                    torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                    self.projection_time += time.time() - start_time

        return per_sample_grads

    def _layernorm_grad_from_grad_comp(
        self,
        layer: nn.LayerNorm,
        idx: int,
        grad_pre_activation: Tensor,
        per_sample: bool = True
    ) -> Tensor:
        """
        Compute the gradient for LayerNorm layers

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
                # Use efficient einsum for 3D case
                grad_weight = torch.einsum("ijk,ijk->ik", grad_pre_activation, normalized)
                grad_bias = torch.sum(grad_pre_activation, dim=1)
            else:
                grad_weight = grad_pre_activation * normalized
                grad_bias = grad_pre_activation
        else:
            if is_3d:
                # Use efficient einsum and sum
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
        self.forward_hooks = [None] * len(self.layer_names)
        self.backward_hooks = [None] * len(self.layer_names)