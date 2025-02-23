from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import Tuple

import torch
import torch.nn as nn

from torch import Tensor

from _dattri.func.projection import random_project

class GCLayerNorm(nn.LayerNorm):
    """
    Gradient Component (GC) LayerNorm layer implementation with gradient factor calculation support.
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(GCLayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.name = 'layernorm'
        self.eps = eps
        self.normalized = None
        self.pre_activation = None

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        self.normalized = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            self.pre_activation = self.weight * self.normalized + self.bias
        else:
            self.pre_activation = self.normalized

        return self.pre_activation

    def grad_comp(self, grad_pre_activation: Tensor, per_sample: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Return the components of the gradient of parameters (for both 2D and 3D inputs).

        For LayerNorm, since the parameter is small, we directly return gradients for the weight and bias.

        Args:
            grad_pre_activation: Gradient of loss w.r.t. pre-activation (dL/dx_o)
            per_sample: Whether to maintain per-sample gradients (default: True)

        Returns:
            tuple: (grad_weight, grad_bias)
        """
        if not self.elementwise_affine:
            raise ValueError("LayerNorm must have learnable parameters for per-example gradients.")

        normalized = self.normalized
        is_3d = self.normalized.dim() == 3

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

        if self.projector_grad_comp_1 is not None:
            grad_weight = self.projector_grad_comp_1(grad_weight)
        if self.projector_grad_comp_2 is not None:
            grad_bias = self.projector_grad_comp_2(grad_bias)

        return grad_weight, grad_bias

    def set_projector(self, base_seed: int, proj_dim_1: int, proj_dim_2: int, projector_kwargs_1: dict, projector_kwargs_2: dict):
        """
        Set the projection function for this layer.

        Args:
            base_seed (int): Base seed for the random projection
            proj_dim_1 (int): Dimension of the projection for the weight
            proj_dim_2 (int): Dimension of the projection for the bias
            projector_kwargs_1 (dict): Keyword arguments for the projection function for the weight
            projector_kwargs_2 (dict): Keyword arguments for the projection function for the bias
        """
        dumb_grad_comp_1 = torch.zeros((self.normalized.shape[0], self.normalized.shape[-1]))
        self.projector_grad_comp_1 = random_project(
            dumb_grad_comp_1,
            dumb_grad_comp_1.shape[0],
            proj_seed=base_seed,
            proj_dim=proj_dim_1,
            **projector_kwargs_1,
        )

        dumb_grad_comp_2 = torch.zeros((self.normalized.shape[0], self.normalized.shape[-1]))
        self.projector_grad_comp_2 = random_project(
            dumb_grad_comp_2,
            dumb_grad_comp_2.shape[0],
            proj_seed=base_seed + 1,
            proj_dim=proj_dim_2,
            **projector_kwargs_2,
        )

    @staticmethod
    def grad_from_grad_comp(grad_weight: Tensor, grad_bias: Tensor) -> Tensor:
        """
        Construct gradient from the gradient components.
        For LayerNorm, components are gradient of weight and bias.

        Args:
            grad_weight: Gradient of loss w.r.t. the weight
            grad_bias: Gradient of loss w.r.t. the bias

        Returns:
            Tensor: Gradient of loss w.r.t. all parameters of the layer
        """
        grad = torch.cat((grad_weight, grad_bias), dim=1)
        return grad

    @staticmethod
    def grad_dot_prod_from_grad_comp(A1: Tensor, B1: Tensor, A2: Tensor, B2: Tensor) -> Tensor:
        """Compute gradient sample norm for the weight matrix in a GCLayerNorm layer.

        Args:
            A1 (Tensor): train weight gradient of the layer.
            B1 (Tensor): train bias gradient of the layer.
            A2 (Tensor): test weight gradient of the layer.
            B2 (Tensor): test bias gradient of the layer.

        Returns:
            Tensor: the gradient sample norm.
        """
        if A1.dim() == 2 and B1.dim() == 2:
            return torch.matmul(A1, A2.T) + torch.matmul(B1, B2.T)
        else:
            raise ValueError(f"Unexpected grad_weight shape: {A1.size()}, grad_bias shape: {B1.size()}")