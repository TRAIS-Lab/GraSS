import torch
import torch.nn as nn
import torch.nn.functional as F

class GCLinear(nn.Linear):
    """
    Gradient Component (GC) Linear layer implementation with gradient factor calculation support.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GCLinear, self).__init__(in_features, out_features, bias)
        self.name = 'linear'
        self.pre_activation = None
        self.layer_input = None
        self.has_bias = bias

    def forward(self, input):
        self.layer_input = input
        out = F.linear(input, self.weight, self.bias)
        self.pre_activation = out

        return self.pre_activation

    def per_sample_grad(self, grad_pre_activation, per_sample=True):
        """
        Return gradient of the pre_activation and the input. Handles both 2D and 3D inputs.

        Args:
            grad_pre_activation: Gradient of the loss w.r.t. the pre-activation (dL/dx_o)
            per_sample: Whether to maintain per-sample gradients (default: True)

        Returns:
            tuple: (grad_pre_activation, augmented_input)
        """
        input_features = self.layer_input
        is_3d = input_features.dim() == 3

        if is_3d:
            batch_size, seq_length, hidden_size = input_features.shape
            # Reshape 3D tensors to 2D for consistent processing
            input_features = input_features.reshape(-1, hidden_size)
            grad_pre_activation = grad_pre_activation.reshape(-1, self.out_features)
        else:
            batch_size = input_features.shape[0]

        # Scale the gradient if we're computing per-sample gradients
        if per_sample:
            grad_pre_activation = grad_pre_activation * batch_size

        # Handle bias term by augmenting input with ones
        if self.has_bias:
            ones = torch.ones(input_features.size(0), 1,
                            device=input_features.device,
                            dtype=input_features.dtype)
            input_features = torch.cat([input_features, ones], dim=1)

        if is_3d:
            # Reshape back to 3D
            input_features = input_features.reshape(batch_size, seq_length, -1)
            grad_pre_activation = grad_pre_activation.reshape(batch_size, seq_length, -1)

        return grad_pre_activation, input_features

class GCEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        # Simply call the parent class's constructor
        super().__init__(*args, **kwargs)

# class GCEmbedding(nn.Embedding):
#     def __init__(self, num_embeddings, embedding_dim):
#         super(GCEmbedding, self).__init__(num_embeddings, embedding_dim)
#         self.pre_activation = None
#         self.indices = None
#         self.name = 'embedding'
#         self.token_output = None
#         self.combined_output = None

#     def forward(self, input):
#         self.indices = input
#         embedded = super().forward(input)
#         self.pre_activation = embedded
#         return embedded

#     def per_sample_grad(self, deriv_pre_activ, per_sample=True):
#         """
#         Prepare components for gradient computation in embedding layer.
#         Similar to linear layer's per_sample_grad but handles sparse embedding lookups.

#         Parameters:
#         -------------------
#         deriv_pre_activ: derivative of cost function w.r.t. the pre-activation of layer
#         per_sample: whether to return per-sample gradients
#         """
#         batch_size = deriv_pre_activ.size(0)

#         # Scale gradients by batch size as in linear layer
#         dLdZ = deriv_pre_activ * batch_size

#         # For sequence inputs (3D)
#         if deriv_pre_activ.dim() == 3:
#             # Create one-hot encoding matrix for the sequence
#             # [batch_size, seq_len, num_embeddings]
#             H = torch.zeros(batch_size, self.indices.size(1), self.num_embeddings,
#                           device=deriv_pre_activ.device)
#             # Fill in ones at the positions indicated by indices
#             H.scatter_(2, self.indices.unsqueeze(-1), 1)
#         else:
#             # For single token inputs (2D)
#             # Create one-hot encoding matrix [batch_size, num_embeddings]
#             H = torch.zeros(batch_size, self.num_embeddings,
#                           device=deriv_pre_activ.device)
#             # Fill in ones at the positions indicated by indices
#             H.scatter_(1, self.indices.unsqueeze(1), 1)

#         print(f"Embedding: {H}", f"Gradient: {dLdZ}")
#         return dLdZ, H