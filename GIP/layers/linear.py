import torch
import torch.nn as nn
import torch.nn.functional as F


class GIPLinear(nn.Linear):
    """LayerNorm implementation with Ghost Inner-Product computation support.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GIPLinear, self).__init__(in_features, out_features, bias)
        self.pre_activation = None
        self.layer_input = None
        self.name = 'linear'
        self.has_bias = bias

    def forward(self, input):
        self.layer_input = input
        out = F.linear(input, self.weight, self.bias)
        self.pre_activation = out

        return self.pre_activation

    def pe_grad_gradcomp(self, output_gradient, per_sample=True):
    # def compute_gradient_terms(self, output_gradient, per_sample=True):
        """
        Compute terms needed for Ghost Inner-Product calculation.

        Handles both 2D and 3D inputs by properly reshaping tensors.

        Args:
            output_gradient: Gradient of loss w.r.t. pre-activation (dL/dx_o)
            per_sample: Whether to maintain per-sample gradients (default: True)

        Returns:
            tuple: (output_gradient, augmented_input)
        """
        input_features = self.layer_input
        is_3d = input_features.dim() == 3

        if is_3d:
            batch_size, seq_length, hidden_size = input_features.shape
            # Reshape 3D tensors to 2D for consistent processing
            input_features = input_features.reshape(-1, hidden_size)
            output_gradient = output_gradient.reshape(-1, self.out_features)
        else:
            batch_size = input_features.shape[0]

        # Scale the gradient if we're computing per-sample gradients
        if per_sample:
            output_gradient = output_gradient * batch_size

        # Handle bias term by augmenting input with ones
        if self.has_bias:
            ones = torch.ones(input_features.size(0), 1,
                            device=input_features.device,
                            dtype=input_features.dtype)
            input_features = torch.cat([input_features, ones], dim=1)

        if is_3d:
            # Reshape back to 3D
            input_features = input_features.reshape(batch_size, seq_length, -1)
            output_gradient = output_gradient.reshape(batch_size, seq_length, -1)

        return output_gradient, input_features

class GIPEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super(GIPEmbedding, self).__init__(num_embeddings, embedding_dim)
        self.pre_activation = None
        self.indices = None
        self.name = 'embedding'
        self.token_output = None
        self.combined_output = None

    def forward(self, input):
        self.indices = input
        embedded = super().forward(input)
        return embedded

    # def store_embedding(self, combined_embedding):
    #     """Store the combined embedding from GPT2's forward pass"""
    #     self.pre_activation = combined_embedding

    # def per_example_gradient(self, deriv_pre_activ):
    #     """
    #     Compute the per-example gradients w.r.t. weights of the layer.

    #     Parameters:
    #     -----------
    #     deriv_pre_activ: derivative of loss function w.r.t. the embedding output
    #     """
    #     batch_size = deriv_pre_activ.size(0)
    #     H = self.indices

    #     if self.indices.dim() == 2: # For single token inputs (2D)
    #         # Scale gradients by batch size
    #         dLdZ = deriv_pre_activ * batch_size

    #         # Compute gradient using batch matrix multiplication [batch_size, num_embeddings, embedding_dim]
    #         pe_grad_weight = torch.bmm(dLdZ.view(batch_size, -1, 1), H.view(batch_size, 1, -1))
    #     else: # For sequence inputs (3D)
    #         dLdZ = deriv_pre_activ.permute(1, 2, 0)
    #         dLdZ *= dLdZ.size(0)
    #         pe_grad_weight = torch.bmm(dLdZ, H.transpose(0, 1))

    #     # Embedding layers don't have bias
    #     # pe_grad_bias = None
    #     return pe_grad_weight

    def per_example_gradient(self, deriv_pre_activ):
        """Compute per-example gradients for the embedding layer"""
        batch_size = deriv_pre_activ.size(0)

        # Use the combined output for position embedding gradients
        if hasattr(self, 'combined_output') and self.combined_output is not None:
            dLdZ = deriv_pre_activ.permute(1, 2, 0)
            dLdZ *= dLdZ.size(0)

            pe_grad_weight = torch.zeros(batch_size, self.num_embeddings, self.embedding_dim,
                                       device=deriv_pre_activ.device)

            for pos in range(self.indices.size(1)):
                pos_indices = self.indices[:, pos]
                pos_grads = dLdZ[pos]

                for b in range(batch_size):
                    pe_grad_weight[b, pos_indices[b]] += pos_grads[:, b]
        else:
            # Regular embedding gradients
            dLdZ = deriv_pre_activ * batch_size
            pe_grad_weight = torch.zeros(batch_size, self.num_embeddings, self.embedding_dim,
                                       device=deriv_pre_activ.device)

            for b in range(batch_size):
                pe_grad_weight[b, self.indices[b]] = dLdZ[b]

        return pe_grad_weight, None

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
        Prepare components for gradient computation in embedding layer.
        Similar to linear layer's pe_grad_gradcomp but handles sparse embedding lookups.

        Parameters:
        -------------------
        deriv_pre_activ: derivative of cost function w.r.t. the pre-activation of layer
        per_sample: whether to return per-sample gradients
        """
        batch_size = deriv_pre_activ.size(0)

        # Scale gradients by batch size as in linear layer
        dLdZ = deriv_pre_activ * batch_size

        # For sequence inputs (3D)
        if deriv_pre_activ.dim() == 3:
            # Create one-hot encoding matrix for the sequence
            # [batch_size, seq_len, num_embeddings]
            H = torch.zeros(batch_size, self.indices.size(1), self.num_embeddings,
                          device=deriv_pre_activ.device)
            # Fill in ones at the positions indicated by indices
            H.scatter_(2, self.indices.unsqueeze(-1), 1)
        else:
            # For single token inputs (2D)
            # Create one-hot encoding matrix [batch_size, num_embeddings]
            H = torch.zeros(batch_size, self.num_embeddings,
                          device=deriv_pre_activ.device)
            # Fill in ones at the positions indicated by indices
            H.scatter_(1, self.indices.unsqueeze(1), 1)

        return dLdZ, H