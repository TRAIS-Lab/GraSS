import torch
import torch.nn as nn

class GGLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(GGLayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        # self.weight = nn.Parameter(torch.ones(normalized_shape))
        # self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.layer_input = None
        self.normalized = None
        self.pre_activation = None
        self.name = 'layernorm'

    def forward(self, x):
        """Forward pass storing intermediate values"""
        self.layer_input = x

        # Calculate statistics
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        self.normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Apply affine transformation
        if self.elementwise_affine:
            self.pre_activation = self.weight * self.normalized + self.bias
        else:
            self.pre_activation = self.normalized

        return self.pre_activation

    def per_example_gradient(self, output_gradient, per_sample=True):
        """Calculate per-example gradients for LayerNorm parameters"""
        if not self.elementwise_affine:
            raise ValueError("LayerNorm must have learnable parameters for per-example gradients.")

        is_3d = self.layer_input.dim() == 3

        if per_sample:
            output_gradient = output_gradient * self.layer_input.shape[0]
            if is_3d:
                grad_weight = torch.einsum("ijk,ijk->ik", output_gradient, self.normalized)
                grad_bias = torch.sum(output_gradient, dim=1)
            else:
                # For 2D input (batch_size, hidden_dim)
                grad_weight = output_gradient * self.normalized
                grad_bias = output_gradient
        else:
            if is_3d:
                # Sum over batch and sequence dimensions
                grad_weight = torch.sum(output_gradient * self.normalized, dim=(0, 1))
                grad_bias = torch.sum(output_gradient, dim=(0, 1))
            else:
                # Sum over batch dimension
                grad_weight = torch.sum(output_gradient * self.normalized, dim=0)
                grad_bias = torch.sum(output_gradient, dim=0)

        return grad_weight, grad_bias