"""
Projector container classes for gradient compression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple
if TYPE_CHECKING:
    from typing import Union

import torch
import torch.nn as nn
import logging

from torch import Tensor
from .projection import random_project

# Configure logger
logger = logging.getLogger(__name__)

class BaseContainer:
    """
    Base container for projection functions associated with a layer.
    """
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index
        self.projector_grad = None
        self.projector_grad_comp = (None, None)

class SparsifierContainer(BaseContainer):
    """
    Container for sparsifier functions associated with a layer.
    Used to store sparsifiers without modifying the original layer.
    """
    pass

class ProjectorContainer(BaseContainer):
    """
    Container for projector functions associated with a layer.
    Used to store projectors without modifying the original layer.
    """
    pass

def setup_model_compressors(
    model: nn.Module,
    layer_names: List[str],
    sparsifier_kwargs: Optional[Dict[str, Any]] = None,
    projector_kwargs: Optional[Dict[str, Any]] = None,
    train_dataloader: torch.utils.data.DataLoader = None,
    setting: str = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[List[SparsifierContainer], List[ProjectorContainer]]:
    """
    Sets up sparsifiers and projectors for each layer in the model.

    Args:
        model: The PyTorch model
        layer_names: Names of layers to set projectors for
        sparsifier_kwargs: Keyword arguments for sparsifier configuration (optional)
        projector_kwargs: Keyword arguments for projector configuration (optional)
        train_dataloader: DataLoader for training data (used to get input shapes)
        setting: Setting name for localized projectors/sparsifiers
        device: Device to run the model on

    Returns:
        Tuple of (sparsifiers, projectors) lists, ordered by layer_names
    """
    if not train_dataloader:
        return [], []

    # Initialize containers lists
    sparsifiers = [None] * len(layer_names) if sparsifier_kwargs else []
    projectors = [None] * len(layer_names) if projector_kwargs else []

    if not (sparsifier_kwargs or projector_kwargs):
        return sparsifiers, projectors

    # Create name to index mapping for faster lookup
    name_to_index = {name: idx for idx, name in enumerate(layer_names)}

    # Extract configuration parameters
    sparsifier_seed = sparsifier_kwargs.get('proj_seed', 0) if sparsifier_kwargs else 0
    sparsifier_factorize = sparsifier_kwargs.get("proj_factorize", True) if sparsifier_kwargs else True
    projector_seed = projector_kwargs.get('proj_seed', 0) if projector_kwargs else 0
    projector_factorize = projector_kwargs.get("proj_factorize", True) if projector_kwargs else True

    # Remove parameters that are handled separately
    if sparsifier_kwargs:
        sparsifier_kwargs_copy = sparsifier_kwargs.copy()
        if 'proj_seed' in sparsifier_kwargs_copy:
            sparsifier_kwargs_copy.pop("proj_seed")
        if 'proj_factorize' in sparsifier_kwargs_copy:
            sparsifier_kwargs_copy.pop("proj_factorize")
    else:
        sparsifier_kwargs_copy = {}

    if projector_kwargs:
        projector_kwargs_copy = projector_kwargs.copy()
        if 'proj_seed' in projector_kwargs_copy:
            projector_kwargs_copy.pop("proj_seed")
        if 'proj_factorize' in projector_kwargs_copy:
            projector_kwargs_copy.pop("proj_factorize")
    else:
        projector_kwargs_copy = {}

    # Run a forward pass to initialize model
    logger.info("Running forward pass to initialize model for compressor setup")
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

    # Create sparsifiers and projectors for each layer
    for module_id, (module_name, module) in enumerate(model.named_modules()):
        if module_name in layer_names:
            idx = name_to_index[module_name]

            # Create sparsifier if configured
            sparsifier = None
            if sparsifier_kwargs:
                sparsifier = SparsifierContainer(module_name, idx)
                base_seed = sparsifier_seed + int(1e4) * module_id

                # Handle special case for localized sparsifiers
                if sparsifier_kwargs_copy.get("method") == "Localize":
                    active_indices = _get_active_indices(
                        module_name,
                        sparsifier_kwargs_copy.get("proj_dim"),
                        layer_inputs.get(module_name),
                        layer_outputs.get(module_name),
                        setting,
                        device,
                        "Sparsifier"
                    )
                    sparse_kwargs = sparsifier_kwargs_copy.copy()
                    sparse_kwargs["active_indices"] = active_indices
                else:
                    sparse_kwargs = sparsifier_kwargs_copy.copy()
                    sparse_kwargs["active_indices"] = None

                # Create appropriate sparsifiers based on layer type
                if isinstance(module, nn.Linear):
                    _setup_linear_compressor(
                        sparsifier,
                        module,
                        layer_inputs.get(module_name),
                        layer_outputs.get(module_name),
                        base_seed,
                        sparse_kwargs,
                        sparsifier_factorize
                    )
                elif isinstance(module, nn.LayerNorm):
                    _setup_layernorm_compressor(
                        sparsifier,
                        module,
                        layer_inputs.get(module_name),
                        layer_outputs.get(module_name),
                        base_seed,
                        sparse_kwargs,
                        sparsifier_factorize
                    )
                else:
                    raise ValueError(f"Unsupported layer type: {type(module)}")

                sparsifiers[idx] = sparsifier

            # Create projector if configured
            projector = None
            if projector_kwargs:
                projector = ProjectorContainer(module_name, idx)
                base_seed = projector_seed + int(1e4) * module_id + 1

                # Determine sparsifier configuration
                using_sparsifier_component = (
                    sparsifier and
                    hasattr(sparsifier, 'projector_grad_comp') and
                    sparsifier.projector_grad_comp != (None, None)
                )
                using_sparsifier_full = (
                    sparsifier and
                    hasattr(sparsifier, 'projector_grad') and
                    sparsifier.projector_grad is not None
                )

                # Check for invalid configuration: sparsifier is full and projector is component
                if using_sparsifier_full and projector_factorize:
                    raise ValueError(
                        f"Invalid configuration for layer {module_name}: Cannot use component projector with full sparsifier."
                    )

                # Handle special case for localized projectors
                if projector_kwargs_copy.get("method") == "Localize":
                    active_indices = _get_active_indices(
                        module_name,
                        projector_kwargs_copy.get("proj_dim"),
                        layer_inputs.get(module_name),
                        layer_outputs.get(module_name),
                        setting,
                        device,
                        "Projector"
                    )
                    proj_kwargs = projector_kwargs_copy.copy()
                    proj_kwargs["active_indices"] = active_indices
                else:
                    proj_kwargs = projector_kwargs_copy.copy()
                    proj_kwargs["active_indices"] = None

                # Set up projectors based on sparsifier configuration
                if using_sparsifier_component:
                    # Create appropriate projectors for sparsified dimensions
                    if isinstance(module, nn.Linear):
                        _setup_linear_compressor_after_sparse(
                            projector,
                            sparsifier,
                            module,
                            layer_inputs.get(module_name),
                            layer_outputs.get(module_name),
                            base_seed,
                            proj_kwargs,
                            projector_factorize
                        )
                    elif isinstance(module, nn.LayerNorm):
                        _setup_layernorm_compressor_after_sparse(
                            projector,
                            sparsifier,
                            module,
                            layer_inputs.get(module_name),
                            layer_outputs.get(module_name),
                            base_seed,
                            proj_kwargs,
                            projector_factorize
                        )
                else:
                    # Create appropriate projectors for original dimensions
                    if isinstance(module, nn.Linear):
                        _setup_linear_compressor(
                            projector,
                            module,
                            layer_inputs.get(module_name),
                            layer_outputs.get(module_name),
                            base_seed,
                            proj_kwargs,
                            projector_factorize
                        )
                    elif isinstance(module, nn.LayerNorm):
                        _setup_layernorm_compressor(
                            projector,
                            module,
                            layer_inputs.get(module_name),
                            layer_outputs.get(module_name),
                            base_seed,
                            proj_kwargs,
                            projector_factorize
                        )

                projectors[idx] = projector

    logger.info(f"Set up compressors for {len(layer_names)} layers")

    return sparsifiers, projectors

def _get_active_indices(
    module_name: str,
    dim: int,
    input_tensor: Tensor,
    output_tensor: Tensor,
    setting: str,
    device: str,
    compressor_type: str
) -> Dict[str, Tensor]:
    """
    Helper function to get active indices for localized compressors
    """
    active_indices = None
    try:
        mask_path = f"../{setting}/Localize/mask_{dim}*{dim}/{module_name}.pt"
        active_indices = torch.load(mask_path, weights_only=False)
        logger.debug(f"Loaded active indices for {module_name} from {mask_path}")
    except FileNotFoundError:
        logger.warning(f"Mask file not found for {module_name} {compressor_type}. Random indices are used.")
        active_indices = {
            "pre_activation": torch.randperm(output_tensor.shape[1])[:dim].to(device),
            "input_features": torch.randperm(input_tensor.shape[1])[:dim].to(device)
        }
    return active_indices


def _setup_linear_compressor(
    container: Union[SparsifierContainer, ProjectorContainer],
    layer: nn.Linear,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any],
    factorize: bool = True
) -> None:
    """
    Set up compressor (sparsifier or projector) for a Linear layer

    Args:
        container: Container to store the compressor functions
        layer: Linear layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        kwargs: Keyword arguments for the projection
        factorize: Whether to factorize the projection
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

    if factorize:
        dumb_grad_comp_1 = torch.zeros_like(pre_activation.view(-1, pre_activation.shape[-1]))
        active_indices = kwargs.get("active_indices", None)
        kwargs_no_indices = kwargs.copy()
        if "active_indices" in kwargs_no_indices:
            kwargs_no_indices.pop("active_indices")

        if active_indices is None:
            active_indices = {"pre_activation": None, "input_features": None}

        projector_grad_comp_1 = random_project(
            dumb_grad_comp_1,
            dumb_grad_comp_1.shape[0],
            proj_seed=base_seed,
            pre_compute=False,
            active_indices=active_indices.get("pre_activation"),
            **kwargs_no_indices
        )

        dumb_grad_comp_2 = torch.zeros_like(input_features.view(-1, input_features.shape[-1]))
        projector_grad_comp_2 = random_project(
            dumb_grad_comp_2,
            dumb_grad_comp_2.shape[0],
            proj_seed=base_seed + 1,
            pre_compute=False,
            active_indices=active_indices.get("input_features"),
            **kwargs_no_indices
        )

        container.projector_grad_comp = (
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
            **kwargs
        )

        container.projector_grad = torch.compile(projector_grad)


def _setup_layernorm_compressor(
    container: Union[SparsifierContainer, ProjectorContainer],
    layer: nn.LayerNorm,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any],
    factorize: bool = True
) -> None:
    """
    Set up compressor (sparsifier or projector) for a LayerNorm layer

    Args:
        container: Container to store the compressor functions
        layer: LayerNorm layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        kwargs: Keyword arguments for the projection
        factorize: Whether to factorize the projection
    """
    if not layer.elementwise_affine:
        return

    if pre_activation is None:
        return

    if factorize:
        dumb_grad_comp_1 = torch.zeros((pre_activation.shape[0], pre_activation.shape[-1]))
        projector_grad_comp_1 = random_project(
            dumb_grad_comp_1,
            dumb_grad_comp_1.shape[0],
            proj_seed=base_seed,
            pre_compute=False,
            **kwargs
        )

        dumb_grad_comp_2 = torch.zeros((pre_activation.shape[0], pre_activation.shape[-1]))
        projector_grad_comp_2 = random_project(
            dumb_grad_comp_2,
            dumb_grad_comp_2.shape[0],
            proj_seed=base_seed + 1,
            pre_compute=False,
            **kwargs
        )

        container.projector_grad_comp = (
            torch.compile(projector_grad_comp_1),
            torch.compile(projector_grad_comp_2)
        )
    else:
        dumb_grad_comp = torch.zeros((pre_activation.shape[0], pre_activation.shape[-1] * 2))
        projector_grad = random_project(
            dumb_grad_comp,
            dumb_grad_comp.shape[0],
            proj_seed=base_seed,
            pre_compute=False,
            **kwargs
        )

        container.projector_grad = torch.compile(projector_grad)


def _setup_linear_compressor_after_sparse(
    projector: ProjectorContainer,
    sparsifier: SparsifierContainer,
    layer: nn.Linear,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any],
    factorize: bool = True
) -> None:
    """
    Set up projector for a Linear layer after sparsification

    Args:
        projector: ProjectorContainer to store the projectors
        sparsifier: SparsifierContainer with sparsification functions
        layer: Linear layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        kwargs: Keyword arguments for the projection
        factorize: Whether to factorize the projection
    """
    if pre_activation is None or layer_input is None:
        return

    batch_size = pre_activation.shape[0]
    is_3d = layer_input.dim() == 3

    # Extract sparsifier components
    sparsifier_grad_comp_1, sparsifier_grad_comp_2 = sparsifier.projector_grad_comp

    # Get sample tensors to determine output dimensions of sparsifiers
    if is_3d:
        sample_pre_activation = pre_activation[:1, :1].reshape(-1, pre_activation.shape[-1])
        sparse_sample_pre_activation = sparsifier_grad_comp_1(sample_pre_activation)
        sparsified_output_dim = sparse_sample_pre_activation.shape[-1]

        input_features = layer_input
        if layer.bias is not None:
            batch_size, seq_length, hidden_size = input_features.shape
            input_features_with_bias = torch.cat([
                input_features,
                torch.ones(batch_size, seq_length, 1, device=input_features.device, dtype=input_features.dtype)
            ], dim=2)
        else:
            input_features_with_bias = input_features

        sample_input_features = input_features_with_bias[:1, :1].reshape(-1, input_features_with_bias.shape[-1])
        sparse_sample_input_features = sparsifier_grad_comp_2(sample_input_features)
        sparsified_input_dim = sparse_sample_input_features.shape[-1]
    else:
        sample_pre_activation = pre_activation[:1]
        sparse_sample_pre_activation = sparsifier_grad_comp_1(sample_pre_activation)
        sparsified_output_dim = sparse_sample_pre_activation.shape[-1]

        input_features = layer_input
        if layer.bias is not None:
            input_features_with_bias = torch.cat([
                input_features,
                torch.ones(input_features.size(0), 1, device=input_features.device, dtype=input_features.dtype)
            ], dim=1)
        else:
            input_features_with_bias = input_features

        sample_input_features = input_features_with_bias[:1]
        sparse_sample_input_features = sparsifier_grad_comp_2(sample_input_features)
        sparsified_input_dim = sparse_sample_input_features.shape[-1]

    if factorize:
        # For component mode projector after component mode sparsifier
        # We need to create projectors that operate on the sparsified dimensions
        # Create dummy tensors with sparsified dimensions
        dumb_grad_comp_1 = torch.zeros(
            (pre_activation.shape[0] if not is_3d else pre_activation.shape[0] * pre_activation.shape[1],
             sparsified_output_dim),
            device=pre_activation.device,
            dtype=pre_activation.dtype
        )

        active_indices = kwargs.get("active_indices", None)
        kwargs_no_indices = kwargs.copy()
        if "active_indices" in kwargs_no_indices:
            kwargs_no_indices.pop("active_indices")

        if active_indices is None:
            active_indices = {"pre_activation": None, "input_features": None}

        projector_grad_comp_1 = random_project(
            dumb_grad_comp_1,
            dumb_grad_comp_1.shape[0],
            proj_seed=base_seed,
            pre_compute=True,
            active_indices=active_indices.get("pre_activation"),
            **kwargs_no_indices
        )

        dumb_grad_comp_2 = torch.zeros(
            (input_features_with_bias.shape[0] if not is_3d else
             input_features_with_bias.shape[0] * input_features_with_bias.shape[1],
             sparsified_input_dim),
            device=input_features.device,
            dtype=input_features.dtype
        )

        projector_grad_comp_2 = random_project(
            dumb_grad_comp_2,
            dumb_grad_comp_2.shape[0],
            proj_seed=base_seed + 1,
            pre_compute=True,
            active_indices=active_indices.get("input_features"),
            **kwargs_no_indices
        )

        projector.projector_grad_comp = (
            torch.compile(projector_grad_comp_1),
            torch.compile(projector_grad_comp_2)
        )
    else:
        # For full projector after component mode sparsifier
        # We need to estimate the dimension of the gradient after applying sparsifiers

        # Calculate outer product dimension after sparsification
        gradient_dim = sparsified_output_dim * sparsified_input_dim

        dumb_grad_full = torch.zeros(
            (batch_size, gradient_dim),
            device=pre_activation.device,
            dtype=pre_activation.dtype
        )

        projector_grad = random_project(
            dumb_grad_full,
            dumb_grad_full.shape[0],
            proj_seed=base_seed,
            pre_compute=True,
            **kwargs
        )

        projector.projector_grad = torch.compile(projector_grad)


def _setup_layernorm_compressor_after_sparse(
    projector: ProjectorContainer,
    sparsifier: SparsifierContainer,
    layer: nn.LayerNorm,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any],
    factorize: bool = True
) -> None:
    """
    Set up projector for a LayerNorm layer after sparsification

    Args:
        projector: ProjectorContainer to store the projectors
        sparsifier: SparsifierContainer with sparsification functions
        layer: LayerNorm layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        kwargs: Keyword arguments for the projection
        factorize: Whether to factorize the projection
    """
    if not layer.elementwise_affine:
        return

    if pre_activation is None:
        return

    # Apply sparsification to get correct dimensions for projector setup
    sparsifier_grad_comp_1, sparsifier_grad_comp_2 = sparsifier.projector_grad_comp

    # Get sample tensors to determine output dimensions of sparsifiers
    sample_pre_activation = pre_activation[:1]
    sparse_sample_pre_activation = sparsifier_grad_comp_1(sample_pre_activation)

    if factorize:
        # Create dummy tensors with sparsified dimensions
        dumb_grad_comp_1 = torch.zeros(
            (pre_activation.shape[0], sparse_sample_pre_activation.shape[-1]),
            device=pre_activation.device,
            dtype=pre_activation.dtype
        )
        projector_grad_comp_1 = random_project(
            dumb_grad_comp_1,
            dumb_grad_comp_1.shape[0],
            proj_seed=base_seed,
            pre_compute=False,
            **kwargs
        )

        dumb_grad_comp_2 = torch.zeros(
            (pre_activation.shape[0], sparse_sample_pre_activation.shape[-1]),
            device=pre_activation.device,
            dtype=pre_activation.dtype
        )
        projector_grad_comp_2 = random_project(
            dumb_grad_comp_2,
            dumb_grad_comp_2.shape[0],
            proj_seed=base_seed + 1,
            pre_compute=False,
            **kwargs
        )

        projector.projector_grad_comp = (
            torch.compile(projector_grad_comp_1),
            torch.compile(projector_grad_comp_2)
        )
    else:
        dumb_grad_comp = torch.zeros((pre_activation.shape[0], sparse_sample_pre_activation.shape[-1] * 2))
        projector_grad = random_project(
            dumb_grad_comp,
            dumb_grad_comp.shape[0],
            proj_seed=base_seed,
            pre_compute=False,
            **kwargs
        )

        projector.projector_grad = torch.compile(projector_grad)