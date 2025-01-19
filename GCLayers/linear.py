import torch
import torch.nn as nn
import torch.nn.functional as F


class GCLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(GCLinear, self).__init__(in_features, out_features, bias)
        self.pre_activation = None
        self.layer_input = None
        self.name = 'linear'
        self.has_bias = bias

    def forward(self, input):
        self.layer_input = input
        out = F.linear(input, self.weight, self.bias)
        self.pre_activation = out

        return self.pre_activation

    def per_example_gradient(self, deriv_pre_activ):
        """
        Computes the per-example gradients w.r.t. weights and bias of the layer.

        Parameters:
        -------------------
        deriv_pre_activ: a tensor containing the derivative of loss function
                         w.r.t. the pre-activation of layer
        """
        is_2d = self.layer_input.dim() == 2
        H = self.layer_input

        if is_2d:
            batch_size = deriv_pre_activ.size(0)
            dLdZ = deriv_pre_activ * batch_size

            pe_grad_weight = torch.bmm(dLdZ.view(batch_size, -1, 1), H.view(batch_size, 1, -1))
            pe_grad_bias = dLdZ
        else:
            dLdZ = deriv_pre_activ.permute(1, 2, 0)
            dLdZ *= dLdZ.size(0)
            pe_grad_weight = torch.bmm(dLdZ, H.transpose(0, 1))
            pe_grad_bias = dLdZ.sum(dim=-1)

        return pe_grad_weight, pe_grad_bias

    def pe_grad_sqnorm(self, deriv_pre_activ):
        """
        Parameters:
        -------------------
        deriv_pre_activ: derivative of cost function w.r.t. the pre-activation of layer
        """
        is_2d = self.layer_input.dim() == 2
        H = self.layer_input

        if is_2d:
            # When the input is a vector, we can compute the per-example gradient,
            # the norm of per-example graidents can be directly computed without materializing them.
            batch_size = deriv_pre_activ.size(0)
            dLdZ = deriv_pre_activ * batch_size

            zsum = dLdZ.pow(2).sum(1)
            hsum = H.pow(2).sum(1)
            s = zsum * hsum

            return s + zsum
        else:
            pe_grad_weight, pe_grad_bias = self.per_example_gradient(deriv_pre_activ)
            batch_size = pe_grad_weight.size(0)
            sq_norm_weight = pe_grad_weight.pow(2).view(batch_size, -1).sum(1)
            sq_norm_bias = pe_grad_bias.pow(2).view(batch_size, -1).sum(1)

            return sq_norm_weight + sq_norm_bias


    def pe_grad_gradcomp(self, deriv_pre_activ, per_sample=True):
        """
        Parameters:
        -------------------
        deriv_pre_activ: derivative of cost function w.r.t. the pre-activation of layer
        """
        is_2d = self.layer_input.dim() == 2
        H = self.layer_input
        batch_size = deriv_pre_activ.shape[0]
        dLdZ = deriv_pre_activ * batch_size

        if is_2d and self.has_bias:
            # Create a column of ones of shape [1000, 1]
            ones_column = torch.ones(H.size(0), 1, device=H.device)
            # Concatenate the column of ones to H along the second dimension (columns)
            H = torch.cat((H, ones_column), dim=1)

        return dLdZ, H

class GCEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super(GCEmbedding, self).__init__(num_embeddings, embedding_dim)
        self.pre_activation = None
        self.indices = None
        self.name = 'embedding'
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, input):
        self.indices = input
        embedded = super().forward(input)
        self.pre_activation = embedded

        return self.pre_activation

    def per_example_gradient(self, deriv_pre_activ):
        """
        Compute the per-example gradients w.r.t. weights of the layer.

        Parameters:
        -----------
        deriv_pre_activ: derivative of loss function w.r.t. the embedding output
        """
        batch_size = deriv_pre_activ.size(0)
        H = self.indices

        if self.indices.dim() == 2: # For single token inputs (2D)
            # Scale gradients by batch size
            dLdZ = deriv_pre_activ * batch_size

            # Compute gradient using batch matrix multiplication [batch_size, num_embeddings, embedding_dim]
            pe_grad_weight = torch.bmm(dLdZ.view(batch_size, -1, 1), H.view(batch_size, 1, -1))
            pe_grad_bias = dLdZ
        else: # For sequence inputs (3D)
            dLdZ = deriv_pre_activ.permute(1, 2, 0)
            dLdZ *= dLdZ.size(0)
            pe_grad_weight = torch.bmm(dLdZ, H.transpose(0, 1))
            pe_grad_bias = dLdZ.sum(dim=-1)

        # Embedding layers don't have bias
        # pe_grad_bias = None
        return pe_grad_weight, pe_grad_bias

    def pe_grad_sqnorm(self, deriv_pre_activ):
        """
        Compute squared norm of per-example gradients for the embedding layer.

        Parameters:
        -----------
        deriv_pre_activ: derivative of loss function w.r.t. the embedding output
        """
        batch_size = deriv_pre_activ.size(0)

        if self.indices.dim() == 2: # For single token inputs (2D)
            dLdZ = deriv_pre_activ * batch_size
            zsum = dLdZ.pow(2).sum(1)
            hsum = self.indices.pow(2).sum(1)
            sq_norms = zsum * hsum + zsum
        else: # For sequence inputs (3D)
            pe_grad_weight, pe_grad_bias = self.per_example_gradient(deriv_pre_activ)
            sq_norm_weight = pe_grad_weight.pow(2).view(batch_size, -1).sum(1)
            sq_norm_bias = pe_grad_bias.pow(2).view(batch_size, -1).sum(1)
            sq_norms = sq_norm_weight + sq_norm_bias

        return sq_norms

    def pe_grad_gradcomp(self, deriv_pre_activ, per_sample=True):
        """
        Prepare components for gradient computation in the embedding layer.

        Parameters:
        -----------
        deriv_pre_activ: derivative of loss function w.r.t. the embedding output
        per_sample: whether to return per-sample gradients
        """
        batch_size = deriv_pre_activ.size(0)
        dLdZ = deriv_pre_activ * batch_size

        return dLdZ, self.indices