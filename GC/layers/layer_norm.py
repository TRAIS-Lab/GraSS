import torch
import torch.nn as nn

from torch import Tensor

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

    def per_sample_grad(self, grad_pre_activation, per_sample=True):
        """
        Compute gradients for the weight and bias. Handles both 2D and 3D inputs.

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

        return grad_weight, grad_bias

    def grad_dot_prod(self, A1: Tensor, B1: Tensor, A2: Tensor, B2: Tensor) -> Tensor:
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