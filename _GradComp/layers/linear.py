from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from ..projection import random_project

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

        self.projector_grad = None
        self.projector_grad_comp = (None, None)

    def forward(self, input):
        self.layer_input = input
        out = F.linear(input, self.weight, self.bias)
        self.pre_activation = out

        return self.pre_activation

    def set_projector(self, base_seed: int, projector_kwargs: dict, proj_factorize: bool = True):
        """
        Set the projection function for this layer.

        Args:
            base_seed (int): Base seed for the random projection
            projector_kwargs (dict): Keyword arguments for the projection function
            proj_factorize (bool): Whether to factorize the projection into two separate projectors
        """
        if self.pre_activation is None or self.layer_input is None:
            raise ValueError("Layer input and pre-activation must be set before setting projectors.")

        batch_size = self.pre_activation.shape[0]

        is_3d = self.layer_input.dim() == 3

        input_features = self.layer_input
        if self.has_bias:
            if is_3d:
                batch_size, seq_length, hidden_size = input_features.shape
                input_features = input_features.reshape(-1, hidden_size)
            else:
                batch_size = input_features.shape[0]

            ones = torch.ones(input_features.size(0), 1, device=input_features.device, dtype=input_features.dtype)
            input_features = torch.cat([input_features, ones], dim=1)

            if is_3d:
                input_features = input_features.reshape(batch_size, seq_length, -1)

        if proj_factorize:
            dumb_grad_comp_1 = torch.zeros_like(self.pre_activation.view(-1, self.pre_activation.shape[-1]))
            active_indices = projector_kwargs.get("active_indices", -1)
            projector_kwargs.pop("active_indices")

            if active_indices == None:
                active_indices = {"pre_activation": None, "input_features": None}

            projector_grad_comp_1 = random_project(
                dumb_grad_comp_1,
                dumb_grad_comp_1.shape[0],
                proj_seed=base_seed,
                pre_compute=proj_factorize,
                active_indices=active_indices["pre_activation"],
                **projector_kwargs,
            )

            dumb_grad_comp_2 = torch.zeros_like(input_features.view(-1, input_features.shape[-1]))
            projector_grad_comp_2 = random_project(
                dumb_grad_comp_2,
                dumb_grad_comp_2.shape[0],
                proj_seed=base_seed + 1,
                pre_compute=proj_factorize,
                active_indices=active_indices["input_features"],
                **projector_kwargs,
            )

            projector_grad_comp_1 = torch.compile(projector_grad_comp_1)
            projector_grad_comp_2 = torch.compile(projector_grad_comp_2)

            self.projector_grad_comp = (projector_grad_comp_1, projector_grad_comp_2)
        else:
            if is_3d:
                dumb_grad = torch.einsum('ijk,ijl->ikl', self.pre_activation, input_features).reshape(batch_size, -1)
            else:
                dumb_grad = torch.einsum('bi,bj->bij', self.pre_activation, input_features).reshape(batch_size, -1)

            projector_grad = random_project(
                dumb_grad,
                dumb_grad.shape[0],
                proj_seed=base_seed,
                pre_compute=proj_factorize,
                **projector_kwargs,
            )

            self.projector_grad = torch.compile(projector_grad)

    # def grad_comp(self, grad_pre_activation: Tensor, per_sample: bool = True) -> Tuple[Tensor, Tensor]:
    #     """
    #     Return the components of the gradient of parameters (for both 2D and 3D inputs).
    #     If projector is set, apply the projector to the gradient components.

    #     For Linear, gradient can be decomposed into the gradient of the pre_activation and the input.

    #     Args:
    #         grad_pre_activation: Gradient of the loss w.r.t. the pre-activation (dL/dx_o)
    #         per_sample: Whether to maintain per-sample gradients (default: True)

    #     Returns:
    #         tuple: (grad_pre_activation, augmented_input)
    #     """
    #     input_features = self.layer_input
    #     is_3d = input_features.dim() == 3

    #     if is_3d:
    #         batch_size, seq_length, hidden_size = input_features.shape
    #         # Reshape 3D tensors to 2D for consistent processing
    #         input_features = input_features.reshape(-1, hidden_size)
    #         grad_pre_activation = grad_pre_activation.reshape(-1, self.out_features)
    #     else:
    #         batch_size = input_features.shape[0]

    #     # Scale the gradient if we're computing per-sample gradients
    #     if per_sample:
    #         grad_pre_activation = grad_pre_activation * batch_size

    #     # Handle bias term by augmenting input with ones
    #     if self.has_bias:
    #         ones = torch.ones(input_features.size(0), 1,
    #                         device=input_features.device,
    #                         dtype=input_features.dtype)
    #         input_features = torch.cat([input_features, ones], dim=1)

    #     if is_3d:
    #         # Reshape back to 3D
    #         input_features = input_features.reshape(batch_size, seq_length, -1)
    #         grad_pre_activation = grad_pre_activation.reshape(batch_size, seq_length, -1)

    #     if self.projector_grad_comp != (None, None):
    #         projector_grad_comp_1, projector_grad_comp_2 = self.projector_grad_comp

    #         grad_pre_activation_flatten = grad_pre_activation.view(-1, grad_pre_activation.shape[-1])
    #         input_features_flatten = input_features.view(-1, input_features.shape[-1])

    #         if is_3d:
    #             grad_pre_activation = projector_grad_comp_1(grad_pre_activation_flatten).view(grad_pre_activation.shape[0], grad_pre_activation.shape[1], -1)
    #             input_features = projector_grad_comp_2(input_features_flatten).view(input_features.shape[0], input_features.shape[1], -1)
    #         else:
    #             grad_pre_activation = projector_grad_comp_1(grad_pre_activation_flatten)
    #             input_features = projector_grad_comp_2(input_features_flatten)

    #     return grad_pre_activation, input_features

    def grad_from_grad_comp(self, grad_pre_activation: Tensor, per_sample: bool = True) -> Tensor:
        """
        Construct gradient from the gradient components.

        For Linear, components are gradient of the pre-activation and the input features.

        Args:
            grad_pre_activation: Gradient of loss w.r.t. the pre-activation

        Returns:
            Tensor: Gradient of loss w.r.t. all parameters of the layer
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

        if self.projector_grad_comp != (None, None):
            projector_grad_comp_1, projector_grad_comp_2 = self.projector_grad_comp

            grad_pre_activation_flatten = grad_pre_activation.view(-1, grad_pre_activation.shape[-1])
            input_features_flatten = input_features.view(-1, input_features.shape[-1])

            if is_3d:
                grad_pre_activation = projector_grad_comp_1(grad_pre_activation_flatten).view(grad_pre_activation.shape[0], grad_pre_activation.shape[1], -1)
                input_features = projector_grad_comp_2(input_features_flatten).view(input_features.shape[0], input_features.shape[1], -1)
            else:
                grad_pre_activation = projector_grad_comp_1(grad_pre_activation_flatten)
                input_features = projector_grad_comp_2(input_features_flatten)

        if is_3d:
            grad = torch.einsum('ijk,ijl->ikl', grad_pre_activation, input_features).reshape(batch_size, -1)
        else:
            grad = torch.einsum('bi,bj->bij', grad_pre_activation, input_features).reshape(batch_size, -1)

        if self.projector_grad != None:
            grad = self.projector_grad(grad)
        return grad

    @staticmethod
    def grad_dot_prod_from_grad_comp(A1: Tensor, B1: Tensor, A2: Tensor, B2: Tensor) -> Tensor:
        """
        Compute gradient sample norm for the weight matrix in a GClinear layer.

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

    def set_projector(self, base_seed: int, projector_kwargs: dict, proj_factorize: bool = True):
        """
        Set the projection function for this layer.

        Args:
            base_seed (int): Base seed for the random projection
            projector_kwargs (dict): Keyword arguments for the projection function
            proj_factorize (bool): Whether to factorize the projection into two separate projectors
        """
        pass