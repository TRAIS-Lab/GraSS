from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple
if TYPE_CHECKING:
    from typing import Union

import torch
import torch.nn as nn

from torch import Tensor
from .projection import random_project

class ProjectorContainer:
    """
    Container for projector functions associated with a layer.
    Used to store projectors without modifying the original layer.
    """
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index
        self.projector_grad = None
        self.projector_grad_comp = (None, None)


def setup_model_projectors(
    model: nn.Module,
    layer_names: List[str],
    projector_kwargs: Dict[str, Any],
    train_dataloader: torch.utils.data.DataLoader,
    setting: str = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> List[ProjectorContainer]:
    """
    Sets up projectors for each layer in the model.

    Args:
        model: The PyTorch model
        layer_names: Names of layers to set projectors for
        projector_kwargs: Keyword arguments for projector configuration
        train_dataloader: DataLoader for training data (used to get input shapes)
        device: Device to run the model on

    Returns:
        List of projector containers, ordered by layer_names
    """
    if not projector_kwargs:
        return []

    # Extract configuration parameters
    proj_seed = projector_kwargs.get('proj_seed', 0)
    proj_factorize = projector_kwargs.get("proj_factorize", True)
    localization = projector_kwargs.get("localization", 0)
    random = projector_kwargs.get("random", 0)

    # Remove parameters that are handled separately
    kwargs_copy = projector_kwargs.copy()
    if 'proj_seed' in kwargs_copy:
        kwargs_copy.pop("proj_seed")
    if 'proj_factorize' in kwargs_copy:
        kwargs_copy.pop("proj_factorize")
    if 'localization' in kwargs_copy:
        kwargs_copy.pop("localization")
    if 'random' in kwargs_copy:
        kwargs_copy.pop("random")

    # Initialize projector containers list
    projectors = [None] * len(layer_names)

    # Create name to index mapping for faster lookup
    name_to_index = {name: idx for idx, name in enumerate(layer_names)}

    # Run a forward pass to initialize model
    train_batch = next(iter(train_dataloader))
    if isinstance(train_batch, dict):
        inputs = {k: v.to(device) for k, v in train_batch.items()}
        model(**inputs)
    else:
        inputs = train_batch[0].to(device)
        model(inputs)

    # First, capture inputs and outputs for each layer
    layer_inputs = {}
    layer_outputs = {}
    hooks = []

    def capture_hook(name, mod, inp, out):
        layer_inputs[name] = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else inp
        layer_outputs[name] = out

    # Register temporary hooks to capture layer I/O
    for name, module in model.named_modules():
        if name in layer_names:
            hook = module.register_forward_hook(lambda mod, inp, out, n=name: capture_hook(n, mod, inp, out))
            hooks.append(hook)

    # Run another forward pass to capture inputs/outputs
    if isinstance(train_batch, dict):
        model(**inputs)
    else:
        model(inputs)

    # Remove temporary hooks
    for hook in hooks:
        hook.remove()

    # Create projectors for each layer
    for module_id, (module_name, module) in enumerate(model.named_modules()):
        if module_name in layer_names:
            idx = name_to_index[module_name]
            projector = ProjectorContainer(module_name, idx)
            base_seed = proj_seed + int(1e4) * module_id

            # Handle special case for localized projectors
            if kwargs_copy.get("method") == "Localize":
                active_indices = None
                try:
                    dim = kwargs_copy["proj_dim"]
                    mask_path = f"../{setting}/Localize/mask_{dim}*{dim}/{module_name}.pt"
                    active_indices = torch.load(mask_path, weights_only=False)
                    # sort in  the indices to ensure they are in the correct order
                    active_indices = {
                        "pre_activation": torch.sort(active_indices["pre_activation"])[0],
                        "input_features": torch.sort(active_indices["input_features"])[0]
                    }
                except FileNotFoundError:
                    print(f"Mask file not found for {module_name}. Random indices are used.")
                    active_indices = {
                        "pre_activation": torch.sort(torch.randperm(layer_outputs[module_name].shape[1])[:dim])[0].to(device),
                        "input_features": torch.sort(torch.randperm(layer_inputs[module_name].shape[1])[:dim])[0].to(device)
                    }
                proj_kwargs = kwargs_copy.copy()
                proj_kwargs["active_indices"] = active_indices
            elif localization > 0:
                active_indices = None
                try:
                    mask_path = f"../{setting}/Localize/mask_{localization}*{localization}/{module_name}.pt"
                    active_indices = torch.load(mask_path, weights_only=False)
                except FileNotFoundError:
                    print(f"Mask file not found for {module_name}. Random indices are used.")
                    active_indices = {
                        "pre_activation": torch.sort(torch.randperm(layer_outputs[module_name].shape[1])[:localization])[0].to(device),
                        "input_features": torch.sort(torch.randperm(layer_inputs[module_name].shape[1])[:localization])[0].to(device)
                    }
                proj_kwargs = kwargs_copy.copy()
                proj_kwargs["active_indices"] = active_indices
            elif random > 0:
                if proj_factorize:
                    active_indices = {
                        "pre_activation": torch.sort(torch.randperm(layer_outputs[module_name].shape[1])[:random])[0].to(device),
                        "input_features": torch.sort(torch.randperm(layer_inputs[module_name].shape[1])[:random])[0].to(device)
                    }
                else:
                    active_indices = torch.sort(torch.randperm(layer_inputs[module_name].shape[1] * layer_outputs[module_name].shape[1])[:random])[0].to(device)
                proj_kwargs = kwargs_copy.copy()
                proj_kwargs["active_indices"] = active_indices
            else:
                proj_kwargs = kwargs_copy.copy()
                proj_kwargs["active_indices"] = None

            # Create appropriate projectors based on layer type
            if isinstance(module, nn.Linear):
                _setup_linear_projector(
                    projector,
                    module,
                    layer_inputs.get(module_name),
                    layer_outputs.get(module_name),
                    base_seed,
                    proj_kwargs,
                    proj_factorize
                )
            elif isinstance(module, nn.LayerNorm):
                _setup_layernorm_projector(
                    projector,
                    module,
                    layer_inputs.get(module_name),
                    layer_outputs.get(module_name),
                    base_seed,
                    proj_kwargs,
                    proj_factorize
                )
            else:
                raise ValueError(f"Unsupported layer type: {type(module)}")

            # Store the projector in the list at the correct index
            projectors[idx] = projector

    return projectors


def _setup_linear_projector(
    projector: ProjectorContainer,
    layer: nn.Linear,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    projector_kwargs: Dict[str, Any],
    proj_factorize: bool = True
) -> None:
    """
    Set up projectors for a Linear layer

    Args:
        projector: ProjectorContainer to store the projectors
        layer: Linear layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        projector_kwargs: Keyword arguments for the projection
        proj_factorize: Whether to factorize the projection
    """
    if pre_activation is None or layer_input is None:
        return

    batch_size = pre_activation.shape[0]
    is_3d = layer_input.dim() == 3

    input_features = layer_input
    if layer.bias is not None:
        if is_3d:
            batch_size, seq_length, hidden_size = input_features.shape
            input_features = input_features.reshape(-1, hidden_size)
        else:
            batch_size = input_features.shape[0]

        ones = torch.ones(input_features.size(0), 1,
                         device=input_features.device,
                         dtype=input_features.dtype)
        input_features = torch.cat([input_features, ones], dim=1)

        if is_3d:
            input_features = input_features.reshape(batch_size, seq_length, -1)

    if proj_factorize:
        dumb_grad_comp_1 = torch.zeros_like(pre_activation.view(-1, pre_activation.shape[-1]))
        active_indices = projector_kwargs.get("active_indices", None)
        projector_kwargs.pop("active_indices")

        if active_indices is None:
            active_indices = {"pre_activation": None, "input_features": None}

        projector_grad_comp_1 = random_project(
            dumb_grad_comp_1,
            dumb_grad_comp_1.shape[0],
            proj_seed=base_seed,
            pre_compute=False,
            active_indices=active_indices.get("pre_activation"),
            **projector_kwargs
        )

        dumb_grad_comp_2 = torch.zeros_like(input_features.view(-1, input_features.shape[-1]))
        projector_grad_comp_2 = random_project(
            dumb_grad_comp_2,
            dumb_grad_comp_2.shape[0],
            proj_seed=base_seed + 1,
            pre_compute=False,
            active_indices=active_indices.get("input_features"),
            **projector_kwargs
        )

        projector.projector_grad_comp = (
            torch.compile(projector_grad_comp_1),
            torch.compile(projector_grad_comp_2)
        )
    else:
        if is_3d:
            dumb_grad = torch.einsum('ijk,ijl->ikl', pre_activation, input_features).reshape(batch_size, -1)
        else:
            dumb_grad = torch.einsum('bi,bj->bij', pre_activation, input_features).reshape(batch_size, -1)

        projector_grad = random_project(
            dumb_grad,
            dumb_grad.shape[0],
            proj_seed=base_seed,
            pre_compute=False,
            **projector_kwargs
        )

        projector.projector_grad = torch.compile(projector_grad)


def _setup_layernorm_projector(
    projector: ProjectorContainer,
    layer: nn.LayerNorm,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    projector_kwargs: Dict[str, Any],
    proj_factorize: bool = True
) -> None:
    """
    Set up projectors for a LayerNorm layer

    Args:
        projector: ProjectorContainer to store the projectors
        layer: LayerNorm layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        projector_kwargs: Keyword arguments for the projection
        proj_factorize: Whether to factorize the projection
    """
    if not layer.elementwise_affine:
        return

    if pre_activation is None:
        return

    if proj_factorize:
        dumb_grad_comp_1 = torch.zeros((pre_activation.shape[0], pre_activation.shape[-1]))
        projector_grad_comp_1 = random_project(
            dumb_grad_comp_1,
            dumb_grad_comp_1.shape[0],
            proj_seed=base_seed,
            pre_compute=proj_factorize,
            **projector_kwargs
        )

        dumb_grad_comp_2 = torch.zeros((pre_activation.shape[0], pre_activation.shape[-1]))
        projector_grad_comp_2 = random_project(
            dumb_grad_comp_2,
            dumb_grad_comp_2.shape[0],
            proj_seed=base_seed + 1,
            pre_compute=proj_factorize,
            **projector_kwargs
        )

        projector.projector_grad_comp = (
            torch.compile(projector_grad_comp_1),
            torch.compile(projector_grad_comp_2)
        )
    else:
        dumb_grad_comp = torch.zeros((pre_activation.shape[0], pre_activation.shape[-1] * 2))
        projector_grad = random_project(
            dumb_grad_comp,
            dumb_grad_comp.shape[0],
            proj_seed=base_seed,
            pre_compute=proj_factorize,
            **projector_kwargs
        )

        projector.projector_grad = torch.compile(projector_grad)