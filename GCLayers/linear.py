import torch
import torch.nn as nn
import torch.nn.functional as F


class GCLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(GCLinear, self).__init__(in_features, out_features, bias)

        # two elements needed for per-example gradient computation
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
        This function computes the per-example gradients w.r.t.
        weights and bias of the layer.

        Parameters:
        -------------------
        deriv_pre_activ: a tensor containing the derivative of loss function
                         with respect to the pre-activation of layer
        """
        is_2d = self.layer_input.dim() == 2
        H = self.layer_input

        if is_2d:
            batch_size = deriv_pre_activ.size(0)
            dLdZ = deriv_pre_activ * batch_size

            pe_grad_weight = torch.bmm(dLdZ.view(batch_size, -1, 1),
                                       H.view(batch_size, 1, -1))
            pe_grad_bias = dLdZ
        else:
            dLdZ = deriv_pre_activ.permute(1, 2, 0)
            dLdZ *= dLdZ.size(0)
            pe_grad_weight = torch.bmm(dLdZ,
                                       H.transpose(0, 1))
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
        # Store input and output for gradient computation
        self.indices = None
        self.embedded = None
        self.name = 'embedding'

    def forward(self, input):
        self.indices = input
        self.embedded = super().forward(input)
        return self.embedded

    def per_example_gradient(self, deriv_pre_activ):
        """
        Compute per-example gradients for the embedding layer.
        For embedding layer, we only need to consider the indices that were actually used.

        Parameters:
        -----------
        deriv_pre_activ: derivative of loss function w.r.t. the embedding output
        """
        batch_size = deriv_pre_activ.size(0)

        # For sequence inputs (3D)
        if deriv_pre_activ.dim() == 3:
            dLdZ = deriv_pre_activ.permute(1, 2, 0)  # Match the linear layer format
            dLdZ *= dLdZ.size(0)  # Scale by batch size as in linear layer

            # Create sparse gradient tensors for each example
            pe_grad_weight = torch.zeros(batch_size, self.num_embeddings, self.embedding_dim,
                                       device=deriv_pre_activ.device)

            # For each position in sequence
            for pos in range(self.indices.size(1)):
                pos_indices = self.indices[:, pos]
                pos_grads = dLdZ[pos]

                # Accumulate gradients for each example's embedding
                for b in range(batch_size):
                    pe_grad_weight[b, pos_indices[b]] += pos_grads[:, b]

        # For single token inputs (2D)
        else:
            dLdZ = deriv_pre_activ * batch_size
            pe_grad_weight = torch.zeros(batch_size, self.num_embeddings, self.embedding_dim,
                                       device=deriv_pre_activ.device)

            # Accumulate gradients for each example's embedding
            for b in range(batch_size):
                pe_grad_weight[b, self.indices[b]] = dLdZ[b]

        # Embedding layers don't have bias
        pe_grad_bias = None
        return pe_grad_weight, pe_grad_bias

    def pe_grad_sqnorm(self, deriv_pre_activ):
        """
        Compute squared norm of per-example gradients for the embedding layer.

        Parameters:
        -----------
        deriv_pre_activ: derivative of loss function w.r.t. the embedding output
        """
        batch_size = deriv_pre_activ.size(0)

        # For sequence inputs (3D)
        if deriv_pre_activ.dim() == 3:
            dLdZ = deriv_pre_activ.permute(1, 2, 0)
            dLdZ *= dLdZ.size(0)

            # Compute squared norm for each example
            sq_norms = torch.zeros(batch_size, device=deriv_pre_activ.device)

            # For each position in sequence
            for pos in range(self.indices.size(1)):
                pos_grads = dLdZ[pos]
                # Add squared norms of gradients for each example
                sq_norms += torch.sum(pos_grads.pow(2), dim=0)

        # For single token inputs (2D)
        else:
            dLdZ = deriv_pre_activ * batch_size
            sq_norms = torch.sum(dLdZ.pow(2), dim=1)

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

        # Create one-hot encoding for the indices
        H = torch.zeros(self.indices.size(0), self.num_embeddings,
                       device=deriv_pre_activ.device)
        H.scatter_(1, self.indices.unsqueeze(1), 1)

        return dLdZ, H