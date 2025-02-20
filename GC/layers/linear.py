import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

def chunked_matmul(A1: Tensor, A2: Tensor, chunk_size=128) -> Tensor:
    """Chuncked matrix multiplication for memory efficiency.

    Args:
        A1 (Tensor): first input tensor of shape [n1, c1, h1, w1]
        A2 (Tensor): second input tensor of shape [n2, c2, w2, h2]
        chunk_size (int, optional): size of each chunk to be multiplied. Defaults to 128.

    Returns:
        Tensor: result of the matrix multiplication of shape [n1, c2, h1, h2].
    """
    if A1.shape[-1] != A2.shape[-2]:
        raise ValueError(f"Inner dimensions must match for matrix multiplication, got {A1.shape[-1]} and {A2.shape[-2]}")

    # Determine output shape
    n1, c1, h1, w1 = A1.shape
    n2, c2, w2, h2 = A2.shape

    if w1 != w2:
        raise ValueError(f"Inner matrix dimensions must agree, got {w1} and {w2}")

    result = torch.zeros(n1, c2, h1, h2, device=A1.device, dtype=A1.dtype)

    for start in range(0, w1, chunk_size):
        end = min(start + chunk_size, w1)
        A1_chunk = A1[:, :, :, start:end]
        A2_chunk = A2[:, :, start:end, :]

        result += torch.matmul(A1_chunk, A2_chunk)

    return result

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

    def grad_comp(self, grad_pre_activation, per_sample=True):
        """
        Return the components of the gradient of parameters (for both 2D and 3D inputs).

        For Linear, gradient can be decomposed into the gradient of the pre_activation and the input.

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

    @staticmethod
    def grad_from_grad_comp(grad_pre_activation: Tensor, input_features: Tensor) -> Tensor:
        """
        Construct gradient from the gradient components.

        For Linear, components are gradient of the pre-activation and the input features.

        Args:
            grad_pre_activation: Gradient of loss w.r.t. the pre-activation
            input_features: Input features to the layer

        Returns:
            Tensor: Gradient of loss w.r.t. all parameters of the layer
        """
        batch_size = grad_pre_activation.shape[0]
        grad = torch.einsum('ijk,ijl->ikl', grad_pre_activation, input_features).reshape(batch_size, -1)
        return grad

    @staticmethod
    def grad_dot_prod_from_grad_comp(A1: Tensor, B1: Tensor, A2: Tensor, B2: Tensor) -> Tensor:
        """Compute gradient sample norm for the weight matrix in a GClinear layer.

        Args:
            A1 (Tensor): train pre_activation gradient of the layer.
            B1 (Tensor): train input to the layer.
            A2 (Tensor): test pre_activation gradient of the layer.
            B2 (Tensor): test input to the layer.

        Returns:
            Tensor: the gradient sample norm.
        """
        if A1.dim() == 2 and B1.dim() == 2:
            dot_prod_1 = torch.matmul(A1, A2.T)
            dot_prod_2 = torch.matmul(B1, B2.T)
            dot_prod = dot_prod_1*dot_prod_2

            return dot_prod

        elif A1.dim() == 3 and B1.dim() == 3:
            (b, t, p), (_, _, d) = A1.size(), B1.size()
            nval, _, _ = A2.size()

            if 2*b*nval*t**2 < (b+nval)*p*d:
                A2, B2 = A2.transpose(-1, -2), B2.transpose(-1, -2)

                A1_expanded = A1.unsqueeze(1)
                A2_expanded = A2.unsqueeze(0)
                B1_expanded = B1.unsqueeze(1)
                B2_expanded = B2.unsqueeze(0)

                # Memory consumption: 2*b*nval*T^2
                A_dotprod = chunked_matmul(A1_expanded, A2_expanded, chunk_size=4096) # Shape: [b, nval, T, T]
                B_dotprod = chunked_matmul(B1_expanded, B2_expanded, chunk_size=4096) # Shape: [b, nval, T, T]

                result = (A_dotprod * B_dotprod).sum(dim=(2, 3))

                return result
            else:
                # [b, p, T] * [b, T, d]
                A = torch.bmm(B1.permute(0, 2, 1), A1).flatten(start_dim=1) # Shape: [b, p*d]
                B = torch.bmm(B2.permute(0, 2, 1), A2).flatten(start_dim=1) # Shape: [nval, p*d]

                return torch.matmul(A, B.T)
        else:
            raise ValueError(f"Unexpected input shape: {A1.size()}, grad_output shape: {B1.size()}")

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

#     def grad_comp(self, deriv_pre_activ, per_sample=True):
#         """
#         Prepare components for gradient computation in embedding layer.
#         Similar to linear layer's grad_comp but handles sparse embedding lookups.

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