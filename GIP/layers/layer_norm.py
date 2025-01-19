import torch
import torch.nn as nn

class GIPLayerNorm(nn.LayerNorm):
    def init(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().init(normalized_shape, eps, elementwise_affine)
        self.name = 'layer_norm'
        self.pre_activation = None
        self.layer_input = None

    def forward(self, x):
        # Use the standard LayerNorm forward pass
        self.layer_input = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps)
        self.pre_activation = super().forward(x)
        return self.pre_activation

    def pe_grad_gradcomp(self, deriv_pre_activ, per_sample=True):
        """
        Prepare components for gradient computation in the layer norm layer.
        Parameters:
        -----------
        deriv_pre_activ: derivative of loss function w.r.t. the layer norm output
        per_sample: whether to return per-sample gradients
        Returns:
        --------
        dLdZ: scaled and permuted gradient
        H: normalized input that can be used for gradient computation
        """
        # Scale and permute gradient similar to per_example_gradient
        dLdZ = deriv_pre_activ
        dLdZ *= dLdZ.size(0)  # Scale by batch size
        # Prepare the normalized input
        H = self.layer_input  # Transpose to match dLdZ dimensions
        return dLdZ, H