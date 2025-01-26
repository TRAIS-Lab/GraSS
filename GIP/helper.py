import torch
from torch import Tensor

from typing import Optional, List, Dict, Any, Tuple, Union

from .layers.linear import GIPLinear, GIPEmbedding
from .layers.layer_norm import GIPLayerNorm

Conv1DSwitch = [
    '.attn.c_attn.weight',
    '.attn.c_proj.weight',
    '.mlp.c_fc.weight',
    '.mlp.c_proj.weight',
]

def transpose_Conv1D(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if any(pattern in key for pattern in Conv1DSwitch):
            new_state_dict[key] = value.t()
        else:
            new_state_dict[key] = value
    return new_state_dict

def find_GIPlayers(model):
    GIP_layers = []

    for module in model.modules():
        if isinstance(module, GIPLinear) or isinstance(module, GIPLayerNorm) or isinstance(module, GIPEmbedding):
            GIP_layers.append(module)

    return GIP_layers

def setup_projectors(
        projector_kwargs: Dict[str, Any],
        layer_name: List[Union[GIPLinear, GIPEmbedding, GIPLayerNorm]],
        mode: Optional[str] = "default",
    ) -> Tuple[Dict[str, Any], List[int], int]:
    """Setup projection dimensions and seeds for each layer.

    Args:
        projector_kwargs (Dict[str, Any]): projector's arguments.
        layer_name (List[Union[GIPLinear, GIPEmbedding, GIPLayerNorm]]): the list of layers to be projected.
        mode (Optional[str], optional): the data attribution's running mode. Defaults to "default".

    Returns:
        Tuple[Dict[str, Any], List[int], int]: processed projector's arguments, projection dimensions, and projection seed.
    """
    if projector_kwargs is None:
        return None, None, None
    proj_seed = projector_kwargs.get("proj_seed", 0)
    proj_dim = projector_kwargs.get("proj_dim", 512)
    proj_dim_dist = projector_kwargs.get("proj_dim_dist", "uniform")

    projector_kwargs.pop("proj_seed")
    projector_kwargs.pop("proj_dim")
    projector_kwargs.pop("proj_dim_dist")

    layer_dim = []
    for layer in layer_name:
        if isinstance(layer, GIPLinear):
            layer_dim.append(layer.weight.shape[0] * layer.weight.shape[1])
        elif isinstance(layer, GIPEmbedding):
            layer_dim.append(layer.embedding_dim * layer.num_embeddings)
        elif isinstance(layer, GIPLayerNorm):
            layer_dim.append(layer.normalized_shape[0])
        else:
            raise ValueError(f"Layer {layer} is not supported")

    if mode == "default" and proj_dim_dist == "non-uniform":
        total_dim = sum(layer_dim)
        proj_dim = [int(proj_dim * dim / total_dim) for dim in layer_dim]
    elif mode in ["one_run", "iterate"] or (mode == "default" and proj_dim_dist == "uniform"):
        proj_dim = [proj_dim] * len(layer_dim)

    print(f"proj_dim: {proj_dim}")
    return projector_kwargs, proj_dim, proj_seed

def grad_dotprod(A1: Tensor, B1: Tensor, A2: Tensor, B2: Tensor) -> Tensor:
    """Compute gradient sample norm for the weight matrix in a GIPlinear layer.

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