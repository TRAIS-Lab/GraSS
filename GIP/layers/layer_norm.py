import torch
import torch.nn as nn

class GIPLayerNorm(nn.LayerNorm):
    """LayerNorm implementation with Ghost Inner-Product computation support.
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(GIPLayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # Save intermediate values for gradient computation
        self.mean = None
        self.sigma = None
        self.normalized = None
        self.layer_input = None
        self.pre_activation = None
        self.name = 'layernorm'

    def forward(self, input):
        self.layer_input = input

        # Calculate mean and variance
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        sigma = (var + self.eps).sqrt()

        # Save intermediate values
        self.mean = mean
        self.sigma = sigma

        # Normalize input
        normalized = (input - mean) / sigma
        self.normalized = normalized

        # Apply affine transformation if enabled
        if self.elementwise_affine:
            output = normalized * self.weight + self.bias
        else:
            output = normalized

        self.pre_activation = output
        return output

    def pe_grad_gradcomp(self, output_gradient, per_sample=True):
        is_3d = self.layer_input.dim() == 3
        if is_3d:
            batch_size, seq_length, hidden_size = self.layer_input.shape

        # For gamma, we need:
        # - output_gradient: dL/dy
        # - normalized input: (x-μ)/σ (should have approximately unit variance)
        normalized = self.normalized  # This should be (x-μ)/σ

        if per_sample:
            output_gradient = output_gradient * batch_size

        if self.elementwise_affine:
            # For γ (weight): need normalized input
            gamma_grad_terms = (output_gradient, normalized)

            # For β (bias): just need output gradient
            ones = torch.ones_like(output_gradient)
            beta_grad_terms = (output_gradient, ones)

            if is_3d:
                gamma_grad_terms = (
                    gamma_grad_terms[0],  # Already in correct shape
                    gamma_grad_terms[1]   # Already in correct shape
                )
                beta_grad_terms = (
                    beta_grad_terms[0],
                    beta_grad_terms[1]
                )

            return gamma_grad_terms, beta_grad_terms
        else:
            return None