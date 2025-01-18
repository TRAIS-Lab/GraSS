import torch.nn as nn

class GCLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self.pre_activation = None
        self.layer_input = None

    def forward(self, x):
        # Use the standard LayerNorm forward pass
        self.layer_input = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps)
        self.pre_activation = super().forward(x)
        return self.pre_activation

    def per_example_gradient(self, deriv_pre_activ):
        """
        Compute the per-example gradient for the layer norm's parameters.
        Args:
            deriv_pre_activ: The gradient of the loss with respect to the layer's pre-activation output.
        Returns:
            Tuple of per-example gradients for weight and bias.
        """
        dLdZ = deriv_pre_activ.permute(1, 0, 2)  # Permute to (feature_dim, batch_size, ...)
        dLdZ *= dLdZ.size(0)  # Scale by batch size
        Z = self.layer_input

        pe_grad_weight = (dLdZ * Z.transpose(0, 1)).sum(dim=1)
        pe_grad_bias = dLdZ.sum(dim=1)

        return pe_grad_weight, pe_grad_bias

    def pe_grad_sqnorm(self, deriv_pre_activ):
        """
        Compute the squared norm of the per-example gradient.
        Args:
            deriv_pre_activ: The gradient of the loss with respect to the layer's pre-activation output.
        Returns:
            Tensor containing the squared norm of the gradient for each example.
        """
        pe_grad_weight, pe_grad_bias = self.per_example_gradient(deriv_pre_activ)
        return pe_grad_weight.pow(2).sum(dim=1) + pe_grad_bias.pow(2).sum(dim=1)