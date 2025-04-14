from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Any
if TYPE_CHECKING:
    from typing import Union

from torch import Tensor

import functools
import torch.nn as nn

class HookManager:
    """
    Manages hooks for efficient gradient component capturing
    without requiring custom layer implementations.
    """
    def __init__(
            self,
            model: nn.Module,
            layer_names: List[str],
        ) -> None:
        """
        Initialize the hook manager

        Args:
            model: The model to hook
            layer_names: Names of layers to hook
        """
        self.model = model
        self.layer_names = layer_names

        # Create mapping from layer name to index for O(1) lookups
        self.layer_name_to_idx = {name: idx for idx, name in enumerate(layer_names)}

        self.forward_hooks = [None] * len(layer_names)
        self.backward_hooks = [None] * len(layer_names)
        self.inputs = [None] * len(layer_names)
        self.pre_activations = [None] * len(layer_names)
        self.grad_pre_activations = [None] * len(layer_names)
        self.normalized = [None] * len(layer_names)  # For LayerNorm if needed

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks to target layers"""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                idx = self.layer_name_to_idx[name]

                # Use functools.partial to correctly bind parameters to avoid late binding issues
                forward_hook = functools.partial(self._forward_hook_fn, idx)
                backward_hook = functools.partial(self._backward_hook_fn, idx)

                # Register hooks with properly bound parameters
                self.forward_hooks[idx] = module.register_forward_hook(forward_hook)
                self.backward_hooks[idx] = module.register_full_backward_hook(backward_hook)

    def _forward_hook_fn(self, idx: int, mod: nn.Module, inp: Any, out: Any) -> None:
        """
        Forward hook function that captures inputs and pre-activations

        Args:
            idx: Layer index
            mod: Module instance
            inp: Input tensors
            out: Output tensors
        """
        # Store input
        if isinstance(inp, tuple) and len(inp) > 0:
            self.inputs[idx] = inp[0].detach()
        else:
            self.inputs[idx] = inp.detach()

        # Store pre-activation (output)
        self.pre_activations[idx] = out.detach()

        # For LayerNorm, also capture the normalized tensor if needed
        if isinstance(mod, nn.LayerNorm):
            x = inp[0] if isinstance(inp, tuple) else inp
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            normalized = (x - mean) / torch.sqrt(var + mod.eps)
            self.normalized[idx] = normalized.detach()

    def _backward_hook_fn(self, idx: int, mod: nn.Module, grad_input: Any, grad_output: Any) -> None:
        """
        Backward hook function that captures output gradients

        Args:
            idx: Layer index
            mod: Module instance
            grad_input: Gradient w.r.t inputs
            grad_output: Gradient w.r.t outputs
        """
        # Store the gradient of the pre-activation
        self.grad_pre_activations[idx] = grad_output[0].detach()

    def get_gradient_components(self, layer_name: str) -> Optional[Tuple[Tensor, Tensor]]:
        """
        Get gradient components for a specific layer

        Args:
            layer_name: Name of the layer

        Returns:
            Tuple of (gradient of pre-activation, input features) or None if not available
        """
        if layer_name not in self.layer_name_to_idx:
            return None

        idx = self.layer_name_to_idx[layer_name]

        # Check if we have the necessary components
        if (self.grad_pre_activations[idx] is None or
            self.inputs[idx] is None):
            return None

        # Return the raw gradient components without any projection
        return self.grad_pre_activations[idx], self.inputs[idx]

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