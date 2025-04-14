from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from typing import List

import torch
import torch.nn as nn

import functools

class HookManager:
    """
    Manages hooks for efficient gradient component capturing and projection
    """
    def __init__(
            self, model: nn.Module,
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

        # Dictionaries to store hooks and projected gradients
        self.forward_hooks = {}
        self.backward_hooks = {}
        self.projected_grads = {}
        self.inputs = {}
        self.pre_activations = {}

        # Register hooks
        self._register_hooks()

    def _forward_hook_fn(self, name, mod, inp, out):
        """Forward hook function implementation"""
        # Store inputs and pre-activations
        self.inputs[name] = inp[0].detach() if isinstance(inp, tuple) and len(inp) > 0 else inp.detach()

        # For GCLinear and GCLayerNorm, store pre-activation directly
        if hasattr(mod, 'pre_activation'):
            mod.pre_activation = out
            self.pre_activations[name] = out.detach()
        else:
            self.pre_activations[name] = out.detach()

    def _backward_hook_fn(self, name, mod, grad_input, grad_output):
        """Backward hook function implementation"""
        # Get pre-activation gradient
        grad_pre_activation = grad_output[0]

        # Project the gradient immediately
        if hasattr(mod, 'grad_from_grad_comp'):
            # Calculate gradient components
            with torch.no_grad():
                # For GCLinear and GCLayerNorm, use their grad_from_grad_comp method
                grad = mod.grad_from_grad_comp(grad_pre_activation, per_sample=True)

                # Store the projected gradient
                self.projected_grads[name] = grad.detach()

    def _register_hooks(self):
        """Register forward and backward hooks to target layers"""
        for name, module in self.model.named_modules():
            if name in self.layer_names and hasattr(module, 'grad_from_grad_comp'):
                # Use functools.partial to correctly bind parameters to avoid late binding issues
                forward_hook = functools.partial(self._forward_hook_fn, name)
                backward_hook = functools.partial(self._backward_hook_fn, name)

                # Register hooks with properly bound parameters
                self.forward_hooks[name] = module.register_forward_hook(forward_hook)
                self.backward_hooks[name] = module.register_full_backward_hook(backward_hook)

    def get_projected_grads(self):
        """Get all projected gradients"""
        return self.projected_grads

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.forward_hooks.values():
            hook.remove()
        for hook in self.backward_hooks.values():
            hook.remove()
        self.forward_hooks = {}
        self.backward_hooks = {}