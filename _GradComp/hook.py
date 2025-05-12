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

import torch.jit as jit

@jit.script
def compute_weight_gradients_2d(grad_pre_activation: Tensor, input_features: Tensor) -> Tensor:
    """
    Compute weight gradients using outer products for 2D tensors.

    Args:
        grad_pre_activation: Gradient of pre-activation with shape [batch_size, output_dim]
        input_features: Input features with shape [batch_size, input_dim]

    Returns:
        Tensor of shape [batch_size, output_dim * input_dim] containing per-sample gradients
    """
    batch_size = input_features.shape[0]
    output_dim = grad_pre_activation.shape[1]
    input_dim = input_features.shape[1]

    # Using einsum for efficient outer product computation
    # For each sample in the batch, compute outer product of grad_pre_activation and input_features
    # 'bi,bj->bij' means: for each batch element b, compute outer product of vectors i and j
    grad_tensor = torch.einsum('bi,bj->bij', grad_pre_activation, input_features)

    # Reshape to [batch_size, output_dim * input_dim]
    return grad_tensor.reshape(batch_size, output_dim * input_dim)

@jit.script
def compute_weight_gradients_3d(grad_pre_activation: Tensor, input_features: Tensor) -> Tensor:
    """
    Compute weight gradients using outer products for 3D tensors (sequence data).

    Args:
        grad_pre_activation: Gradient of pre-activation with shape [batch_size, seq_length, output_dim]
        input_features: Input features with shape [batch_size, seq_length, input_dim]

    Returns:
        Tensor of shape [batch_size, output_dim * input_dim] containing per-sample gradients
    """
    batch_size = input_features.shape[0]
    seq_length = input_features.shape[1]
    output_dim = grad_pre_activation.shape[2]
    input_dim = input_features.shape[2]

    grad_tensor = torch.einsum('bsi,bsj->bij', grad_pre_activation, input_features)
    # grad_tensor = grad_tensor.sum(dim=1)
    return grad_tensor.reshape(batch_size, output_dim * input_dim)

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
        self.compression_time = 0.0

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

    def set_sparsifiers(self, sparsifiers: List[Any]) -> None:
        """
        Set specifier objects for each layer

        Args:
            sparsifiers: List of projector objects, ordered by layer_names
        """
        self.sparsifiers = sparsifiers

    def set_projectors(self, projectors: List[Any]) -> None:
        """
        Set projector objects for each layer

        Args:
            projectors: List of projector objects, ordered by layer_names
        """
        self.projectors = projectors

    def get_compressed_grads(self) -> List[Tensor]:
        """
        Get all captured projected gradients

        Returns:
            List of projected gradient tensors, ordered by layer_names
        """
        return self.projected_grads

    def get_compression_time(self) -> float:
        """
        Get the accumulated projection time

        Returns:
            Total time spent in projection operations
        """
        return self.compression_time

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
        Compute the gradient for Linear layers with two-stage compression:
        sparsifiers (stage 1) and projectors (stage 2).

        Supports different behaviors with either or both compression methods.

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

        # Get sparsifier and projector for this layer (if they exist)
        sparsifier = self.sparsifiers[idx] if hasattr(self, 'sparsifiers') and idx < len(self.sparsifiers) else None
        projector = self.projectors[idx] if hasattr(self, 'projectors') and idx < len(self.projectors) else None

        # Determine sparsifier configuration
        using_sparsifier_component = (
            sparsifier and
            hasattr(sparsifier, 'projector_grad_comp') and
            sparsifier.projector_grad_comp != (None, None)
        )
        using_sparsifier_full = (
            sparsifier and
            hasattr(sparsifier, 'projector_grad') and
            sparsifier.projector_grad is not None
        )

        # Determine projector configuration
        using_projector_component = (
            projector and
            hasattr(projector, 'projector_grad_comp') and
            projector.projector_grad_comp != (None, None)
        )
        using_projector_full = (
            projector and
            hasattr(projector, 'projector_grad') and
            projector.projector_grad is not None
        )

        # Check for invalid configuration: sparsifier is full and projector is component
        assert not (using_sparsifier_full and using_projector_component), \
            "Cannot use component projector with full sparsifier."

        # Optimized path: When we have no sparsification or projection
        if not using_sparsifier_component and not using_sparsifier_full and not using_projector_component and not using_projector_full:
            # Simply compute the gradient without any compression
            if is_3d:
                batch_size, seq_length, hidden_size = input_features.shape
                input_features_flat = input_features.reshape(-1, hidden_size)
                grad_pre_activation_flat = grad_pre_activation.reshape(-1, layer.out_features)

                if per_sample:
                    grad_pre_activation_flat = grad_pre_activation_flat * batch_size

                if layer.bias is not None:
                    ones = torch.ones(
                        input_features_flat.size(0), 1,
                        device=input_features_flat.device,
                        dtype=input_features_flat.dtype
                    )
                    input_features_flat = torch.cat([input_features_flat, ones], dim=1)

                input_features_3d = input_features_flat.reshape(batch_size, seq_length, -1)
                grad_pre_activation_3d = grad_pre_activation_flat.reshape(batch_size, seq_length, -1)

                grad_tensor = compute_weight_gradients_3d(grad_pre_activation_3d, input_features_3d)
            else:
                if per_sample:
                    grad_pre_activation = grad_pre_activation * grad_pre_activation.shape[0]

                if layer.bias is not None:
                    ones = torch.ones(
                        input_features.size(0), 1,
                        device=input_features.device,
                        dtype=input_features.dtype
                    )
                    input_features = torch.cat([input_features, ones], dim=1)

                grad_tensor = compute_weight_gradients_2d(grad_pre_activation, input_features)

            return grad_tensor.reshape(input_features.shape[0] if not is_3d else batch_size, -1)

        # Optimized path: When we only need to project the final gradient (no sparsification, only full projector)
        if not using_sparsifier_component and not using_sparsifier_full and using_projector_full:
            return self._compute_projected_param_gradients(layer, idx, input_features,
                                                          grad_pre_activation, per_sample)

        # Process tensors for gradient computation
        if is_3d:
            batch_size, seq_length, hidden_size = input_features.shape
            # Reshape 3D tensors to 2D for consistent processing
            input_features_flat = input_features.reshape(-1, hidden_size)
            grad_pre_activation_flat = grad_pre_activation.reshape(-1, layer.out_features)
        else:
            batch_size = input_features.shape[0]
            input_features_flat = input_features
            grad_pre_activation_flat = grad_pre_activation

        # Scale the gradient if we're computing per-sample gradients
        if per_sample:
            grad_pre_activation_flat = grad_pre_activation_flat * batch_size

        # Handle bias term by augmenting input with ones
        if layer.bias is not None:
            ones = torch.ones(
                input_features_flat.size(0), 1,
                device=input_features_flat.device,
                dtype=input_features_flat.dtype
            )
            input_features_flat = torch.cat([input_features_flat, ones], dim=1)

        # Reshape back to 3D if needed
        if is_3d:
            input_features_3d = input_features_flat.reshape(batch_size, seq_length, -1)
            grad_pre_activation_3d = grad_pre_activation_flat.reshape(batch_size, seq_length, -1)

        # Case: Only use sparsification in component mode
        if using_sparsifier_component and not using_projector_component and not using_projector_full:
            # Apply sparsification to gradient components
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                start_time = time.time()

            sparsifier_grad_comp_1, sparsifier_grad_comp_2 = sparsifier.projector_grad_comp

            if is_3d:
                grad_pre_activation_sparse = sparsifier_grad_comp_1(grad_pre_activation_flat).reshape(
                    batch_size, seq_length, -1
                )
                input_features_sparse = sparsifier_grad_comp_2(input_features_flat).reshape(
                    batch_size, seq_length, -1
                )

                # Compute gradient with sparsified components
                grad_tensor = compute_weight_gradients_3d(grad_pre_activation_sparse, input_features_sparse)
            else:
                grad_pre_activation_sparse = sparsifier_grad_comp_1(grad_pre_activation_flat)
                input_features_sparse = sparsifier_grad_comp_2(input_features_flat)

                # Compute gradient with sparsified components
                grad_tensor = compute_weight_gradients_2d(grad_pre_activation_sparse, input_features_sparse)

            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                self.compression_time += time.time() - start_time

            return grad_tensor.reshape(batch_size, -1)

        # Case: Only use projection in component mode
        if not using_sparsifier_component and not using_sparsifier_full and using_projector_component:
            # Apply projection to gradient components
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                start_time = time.time()

            projector_grad_comp_1, projector_grad_comp_2 = projector.projector_grad_comp

            if is_3d:
                grad_pre_activation_proj = projector_grad_comp_1(grad_pre_activation_flat).reshape(
                    batch_size, seq_length, -1
                )
                input_features_proj = projector_grad_comp_2(input_features_flat).reshape(
                    batch_size, seq_length, -1
                )

                # Compute gradient with projected components
                grad_tensor = compute_weight_gradients_3d(grad_pre_activation_proj, input_features_proj)
            else:
                grad_pre_activation_proj = projector_grad_comp_1(grad_pre_activation_flat)
                input_features_proj = projector_grad_comp_2(input_features_flat)

                # Compute gradient with projected components
                grad_tensor = compute_weight_gradients_2d(grad_pre_activation_proj, input_features_proj)

            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                self.compression_time += time.time() - start_time

            return grad_tensor.reshape(batch_size, -1)

        # Case: Only use full sparsification
        if using_sparsifier_full and not using_projector_component and not using_projector_full:
            if is_3d:
                grad_tensor = compute_weight_gradients_3d(grad_pre_activation_3d, input_features_3d)
            else:
                grad_tensor = compute_weight_gradients_2d(grad_pre_activation_flat, input_features_flat)

            grad = grad_tensor.reshape(batch_size, -1)

            # Apply full sparsifier
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                start_time = time.time()

            grad = sparsifier.projector_grad(grad)

            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                self.compression_time += time.time() - start_time

            return grad

        # Stage 1: Apply sparsification in component mode if available
        if using_sparsifier_component:
            # Start timing for sparsification if profiling is enabled
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                start_time = time.time()

            sparsifier_grad_comp_1, sparsifier_grad_comp_2 = sparsifier.projector_grad_comp

            # Apply sparsification to gradient components
            if is_3d:
                grad_pre_activation_sparse = sparsifier_grad_comp_1(grad_pre_activation_flat).reshape(
                    batch_size, seq_length, -1
                )
                input_features_sparse = sparsifier_grad_comp_2(input_features_flat).reshape(
                    batch_size, seq_length, -1
                )
            else:
                grad_pre_activation_sparse = sparsifier_grad_comp_1(grad_pre_activation_flat)
                input_features_sparse = sparsifier_grad_comp_2(input_features_flat)

            # End timing for sparsification
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                self.compression_time += time.time() - start_time

            # Stage 2: Apply projection in component mode if available
            if using_projector_component:
                # Start timing for projection if profiling is enabled
                if self.profile:
                    torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                    start_time = time.time()

                projector_grad_comp_1, projector_grad_comp_2 = projector.projector_grad_comp

                # Apply projection to sparsified components
                if is_3d:
                    grad_pre_activation_sparse_flat = grad_pre_activation_sparse.reshape(-1, grad_pre_activation_sparse.shape[-1])
                    input_features_sparse_flat = input_features_sparse.reshape(-1, input_features_sparse.shape[-1])

                    grad_pre_activation_final = projector_grad_comp_1(grad_pre_activation_sparse_flat).reshape(
                        batch_size, seq_length, -1
                    )
                    input_features_final = projector_grad_comp_2(input_features_sparse_flat).reshape(
                        batch_size, seq_length, -1
                    )
                else:
                    grad_pre_activation_final = projector_grad_comp_1(grad_pre_activation_sparse)
                    input_features_final = projector_grad_comp_2(input_features_sparse)

                # End timing for projection
                if self.profile:
                    torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                    self.compression_time += time.time() - start_time

                # Compute the final gradient using the processed components
                if is_3d:
                    grad_tensor = compute_weight_gradients_3d(grad_pre_activation_final, input_features_final)
                else:
                    grad_tensor = compute_weight_gradients_2d(grad_pre_activation_final, input_features_final)

                grad = grad_tensor.reshape(batch_size, -1)

            else:  # Using sparsifier component with full projector
                # Compute gradient with sparsified components
                if is_3d:
                    grad_tensor = compute_weight_gradients_3d(grad_pre_activation_sparse, input_features_sparse)
                else:
                    grad_tensor = compute_weight_gradients_2d(grad_pre_activation_sparse, input_features_sparse)

                grad = grad_tensor.reshape(batch_size, -1)

                # Apply full projector if available
                if using_projector_full:
                    # Start timing for projection if profiling is enabled
                    if self.profile:
                        torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                        start_time = time.time()

                    grad = projector.projector_grad(grad)

                    # End timing for projection
                    if self.profile:
                        torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                        self.compression_time += time.time() - start_time

        else:  # Not using sparsifier component mode
            # Compute the outer product to get the gradient
            if is_3d:
                grad_tensor = compute_weight_gradients_3d(grad_pre_activation_3d, input_features_3d)
            else:
                grad_tensor = compute_weight_gradients_2d(grad_pre_activation_flat, input_features_flat)

            grad = grad_tensor.reshape(batch_size, -1)

            # Apply full sparsifier if available
            if using_sparsifier_full:
                # Start timing for sparsification if profiling is enabled
                if self.profile:
                    torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                    start_time = time.time()

                grad = sparsifier.projector_grad(grad)

                # End timing for sparsification
                if self.profile:
                    torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                    self.compression_time += time.time() - start_time

            # Apply full projector if available
            if using_projector_full:
                # Start timing for projection if profiling is enabled
                if self.profile:
                    torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                    start_time = time.time()

                grad = projector.projector_grad(grad)

                # End timing for projection
                if self.profile:
                    torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                    self.compression_time += time.time() - start_time

        return grad

    # def _linear_grad_from_grad_comp(
    #     self,
    #     layer: nn.Linear,
    #     idx: int,
    #     grad_pre_activation: Tensor,
    #     per_sample: bool = True
    # ) -> Tensor:
    #     """
    #     Compute the gradient for Linear layers with two-stage compression:
    #     sparsifiers (stage 1) and projectors (stage 2).

    #     Supports four different behaviors:
    #     1. Sparsifier is full and projector is full
    #     2. Sparsifier is full and projector is component (invalid)
    #     3. Sparsifier is component and projector is full
    #     4. Sparsifier is component and projector is component

    #     Args:
    #         layer: Linear layer
    #         idx: Layer index
    #         grad_pre_activation: Gradient of the pre-activation
    #         per_sample: Whether to compute per-sample gradients

    #     Returns:
    #         Projected gradient tensor
    #     """
    #     input_features = self.inputs[idx]
    #     is_3d = input_features.dim() == 3

    #     # Get sparsifier and projector for this layer
    #     sparsifier = self.sparsifiers[idx] if hasattr(self, 'sparsifiers') and idx < len(self.sparsifiers) else None
    #     projector = self.projectors[idx] if hasattr(self, 'projectors') and idx < len(self.projectors) else None

    #     # Determine sparsifier configuration
    #     using_sparsifier_component = (
    #         sparsifier and
    #         hasattr(sparsifier, 'projector_grad_comp') and
    #         sparsifier.projector_grad_comp != (None, None)
    #     )
    #     using_sparsifier_full = (
    #         sparsifier and
    #         hasattr(sparsifier, 'projector_grad') and
    #         sparsifier.projector_grad is not None
    #     )

    #     # Determine projector configuration
    #     using_projector_component = (
    #         projector and
    #         hasattr(projector, 'projector_grad_comp') and
    #         projector.projector_grad_comp != (None, None)
    #     )
    #     using_projector_full = (
    #         projector and
    #         hasattr(projector, 'projector_grad') and
    #         projector.projector_grad is not None
    #     )

    #     print(f"Layer {idx}: using_sparsifier_component={using_sparsifier_component}, "
    #           f"using_sparsifier_full={using_sparsifier_full}, "
    #           f"using_projector_component={using_projector_component}, "
    #           f"using_projector_full={using_projector_full}")

    #     # Check for invalid configuration: sparsifier is full and projector is component
    #     assert not (using_sparsifier_full and using_projector_component), \
    #         "Cannot use component projector with full sparsifier."

    #     # Optimized path: When we only need to project the final gradient and no sparsification
    #     if not using_sparsifier_component and not using_sparsifier_full and using_projector_full:
    #         return self._compute_projected_param_gradients(layer, idx, input_features,
    #                                                       grad_pre_activation, per_sample)

    #     # Process tensors for gradient computation
    #     if is_3d:
    #         batch_size, seq_length, hidden_size = input_features.shape
    #         # Reshape 3D tensors to 2D for consistent processing
    #         input_features_flat = input_features.reshape(-1, hidden_size)
    #         grad_pre_activation_flat = grad_pre_activation.reshape(-1, layer.out_features)
    #     else:
    #         batch_size = input_features.shape[0]
    #         input_features_flat = input_features
    #         grad_pre_activation_flat = grad_pre_activation

    #     # Scale the gradient if we're computing per-sample gradients
    #     if per_sample:
    #         grad_pre_activation_flat = grad_pre_activation_flat * batch_size

    #     # Handle bias term by augmenting input with ones
    #     if layer.bias is not None:
    #         ones = torch.ones(
    #             input_features_flat.size(0), 1,
    #             device=input_features_flat.device,
    #             dtype=input_features_flat.dtype
    #         )
    #         input_features_flat = torch.cat([input_features_flat, ones], dim=1)

    #     # Reshape back to 3D if needed
    #     if is_3d:
    #         input_features_3d = input_features_flat.reshape(batch_size, seq_length, -1)
    #         grad_pre_activation_3d = grad_pre_activation_flat.reshape(batch_size, seq_length, -1)

    #     # Stage 1: Apply sparsification in component mode if available
    #     if using_sparsifier_component:
    #         # Start timing for sparsification if profiling is enabled
    #         if self.profile:
    #             torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
    #             start_time = time.time()

    #         sparsifier_grad_comp_1, sparsifier_grad_comp_2 = sparsifier.projector_grad_comp

    #         # Apply sparsification to gradient components
    #         if is_3d:
    #             grad_pre_activation_sparse = sparsifier_grad_comp_1(grad_pre_activation_flat).reshape(
    #                 batch_size, seq_length, -1
    #             )
    #             input_features_sparse = sparsifier_grad_comp_2(input_features_flat).reshape(
    #                 batch_size, seq_length, -1
    #             )
    #         else:
    #             grad_pre_activation_sparse = sparsifier_grad_comp_1(grad_pre_activation_flat)
    #             input_features_sparse = sparsifier_grad_comp_2(input_features_flat)

    #         # End timing for sparsification
    #         if self.profile:
    #             torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
    #             self.compression_time += time.time() - start_time

    #         # Stage 2: Apply projection in component mode if available
    #         if using_projector_component:
    #             # Start timing for projection if profiling is enabled
    #             if self.profile:
    #                 torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
    #                 start_time = time.time()

    #             projector_grad_comp_1, projector_grad_comp_2 = projector.projector_grad_comp

    #             # Apply projection to sparsified components
    #             if is_3d:
    #                 grad_pre_activation_sparse_flat = grad_pre_activation_sparse.reshape(-1, grad_pre_activation_sparse.shape[-1])
    #                 input_features_sparse_flat = input_features_sparse.reshape(-1, input_features_sparse.shape[-1])

    #                 grad_pre_activation_final = projector_grad_comp_1(grad_pre_activation_sparse_flat).reshape(
    #                     batch_size, seq_length, -1
    #                 )
    #                 input_features_final = projector_grad_comp_2(input_features_sparse_flat).reshape(
    #                     batch_size, seq_length, -1
    #                 )
    #             else:
    #                 grad_pre_activation_final = projector_grad_comp_1(grad_pre_activation_sparse)
    #                 input_features_final = projector_grad_comp_2(input_features_sparse)

    #             # End timing for projection
    #             if self.profile:
    #                 torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
    #                 self.compression_time += time.time() - start_time

    #             # Compute the final gradient using the processed components
    #             if is_3d:
    #                 grad_tensor = compute_weight_gradients_3d(grad_pre_activation_final, input_features_final)
    #             else:
    #                 grad_tensor = compute_weight_gradients_2d(grad_pre_activation_final, input_features_final)

    #             grad = grad_tensor.reshape(batch_size, -1)

    #         else:  # Using sparsifier component with full projector
    #             # Compute gradient with sparsified components
    #             if is_3d:
    #                 grad_tensor = compute_weight_gradients_3d(grad_pre_activation_sparse, input_features_sparse)
    #             else:
    #                 grad_tensor = compute_weight_gradients_2d(grad_pre_activation_sparse, input_features_sparse)

    #             grad = grad_tensor.reshape(batch_size, -1)

    #             # Apply full projector if available
    #             if using_projector_full:
    #                 # Start timing for projection if profiling is enabled
    #                 if self.profile:
    #                     torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
    #                     start_time = time.time()

    #                 grad = projector.projector_grad(grad)

    #                 # End timing for projection
    #                 if self.profile:
    #                     torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
    #                     self.compression_time += time.time() - start_time

    #     else:  # Not using sparsifier component mode
    #         # Compute the outer product to get the gradient
    #         if is_3d:
    #             grad_tensor = compute_weight_gradients_3d(grad_pre_activation_3d, input_features_3d)
    #         else:
    #             grad_tensor = compute_weight_gradients_2d(grad_pre_activation_flat, input_features_flat)

    #         grad = grad_tensor.reshape(batch_size, -1)

    #         # Apply full sparsifier if available
    #         if using_sparsifier_full:
    #             # Start timing for sparsification if profiling is enabled
    #             if self.profile:
    #                 torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
    #                 start_time = time.time()

    #             grad = sparsifier.projector_grad(grad)

    #             # End timing for sparsification
    #             if self.profile:
    #                 torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
    #                 self.compression_time += time.time() - start_time

    #         # Apply full projector if available
    #         if using_projector_full:
    #             # Start timing for projection if profiling is enabled
    #             if self.profile:
    #                 torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
    #                 start_time = time.time()

    #             grad = projector.projector_grad(grad)

    #             # End timing for projection
    #             if self.profile:
    #                 torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
    #                 self.compression_time += time.time() - start_time

    #     return grad

    def _compute_projected_param_gradients(
        self,
        layer: nn.Linear,
        idx: int,
        input_features: Tensor,
        grad_pre_activation: Tensor,
        per_sample: bool = True
    ) -> Tensor:
        """
        Compute per-sample gradients for a linear layer using PyTorch's efficient
        matrix operations and apply projections.

        This method is optimized for the case when only the gradient projector is provided.

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

        # Start timing for the whole operation if profiling is enabled
        if self.profile:
            torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
            start_time = time.time()

        with torch.no_grad():
            # Handle different input dimensions
            is_3d = input_features.dim() == 3

            if is_3d:
                # For 3D inputs (sequence data)
                batch_size, seq_length, hidden_size = input_features.shape

                # Compute weight gradients for each sample in the batch
                # Reshape to 2D, compute gradients, then reshape back
                input_reshaped = input_features.view(-1, hidden_size)
                grad_reshaped = grad_pre_activation.view(-1, layer.out_features)

                # Efficient matrix multiplication for weight gradients
                # Using batched matmul to compute all per-sample gradients at once
                if per_sample:
                    # Scale gradients for per-sample computation
                    grad_reshaped = grad_reshaped * batch_size

                    # Reshape for batched matrix multiplication
                    input_batched = input_reshaped.view(batch_size, seq_length, hidden_size)
                    grad_batched = grad_reshaped.view(batch_size, seq_length, layer.out_features)

                    # Efficient batched computation (B x S x O) @ (B x S x I) -> (B x O x I)
                    # Sum over sequence dimension for each sample
                    per_sample_grads_w = torch.bmm(
                        grad_batched.transpose(1, 2),  # B x O x S
                        input_batched                  # B x S x I
                    )  # Result: B x O x I

                    # For bias gradients, sum over sequence dimension
                    if layer.bias is not None:
                        per_sample_grads_b = grad_batched.sum(dim=1)  # B x O

                        # Combine weight and bias gradients
                        per_sample_grads_w = per_sample_grads_w.view(batch_size, -1)  # B x (O*I)
                        per_sample_grads = torch.cat([per_sample_grads_w, per_sample_grads_b], dim=1)
                    else:
                        per_sample_grads = per_sample_grads_w.view(batch_size, -1)
                else:
                    # For full batch gradient, just do standard matrix multiplication
                    per_sample_grads_w = torch.matmul(grad_reshaped.t(), input_reshaped)  # O x I

                    if layer.bias is not None:
                        per_sample_grads_b = grad_reshaped.sum(dim=0)  # O
                        per_sample_grads = torch.cat([per_sample_grads_w.view(-1), per_sample_grads_b])
                    else:
                        per_sample_grads = per_sample_grads_w.view(-1)

                    # Expand to match batch dimension for consistent return format
                    per_sample_grads = per_sample_grads.expand(batch_size, -1)
            else:
                # For 2D inputs (standard batch data)
                if per_sample:
                    # Scale gradients for per-sample computation
                    grad_pre_activation = grad_pre_activation * batch_size

                    # Efficient per-sample gradient computation
                    # Each sample computed independently using outer product
                    per_sample_grads_w = torch.bmm(
                        grad_pre_activation.unsqueeze(2),   # B x O x 1
                        input_features.unsqueeze(1)         # B x 1 x I
                    )  # Result: B x O x I

                    # For bias gradients
                    if layer.bias is not None:
                        per_sample_grads_b = grad_pre_activation  # B x O

                        # Combine weight and bias gradients
                        per_sample_grads_w = per_sample_grads_w.view(batch_size, -1)  # B x (O*I)
                        per_sample_grads = torch.cat([per_sample_grads_w, per_sample_grads_b], dim=1)
                    else:
                        per_sample_grads = per_sample_grads_w.view(batch_size, -1)
                else:
                    # For full batch gradient
                    per_sample_grads_w = torch.matmul(grad_pre_activation.t(), input_features)  # O x I

                    if layer.bias is not None:
                        per_sample_grads_b = grad_pre_activation.sum(dim=0)  # O
                        per_sample_grads = torch.cat([per_sample_grads_w.view(-1), per_sample_grads_b])
                    else:
                        per_sample_grads = per_sample_grads_w.view(-1)

                    # Expand to match batch dimension for consistent return format
                    per_sample_grads = per_sample_grads.expand(batch_size, -1)

            # Apply gradient projector
            if projector and hasattr(projector, 'projector_grad') and projector.projector_grad is not None:
                per_sample_grads = projector.projector_grad(per_sample_grads)

        # End timing for the whole operation
        if self.profile:
            torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
            self.compression_time += time.time() - start_time

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
                self.compression_time += time.time() - start_time

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
                self.compression_time += time.time() - start_time

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