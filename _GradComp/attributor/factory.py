"""
Factory and utilities for creating attributors.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union, Literal

if TYPE_CHECKING:
    import torch.nn as nn

from .attributor import IFAttributor
from .base import HessianOptions

# Type definitions
OffloadOptions = Literal["none", "cpu", "disk"]

def create_attributor(
    attributor_type: str,
    setting: str,
    model: 'nn.Module',
    layer_names: Union[str, List[str]],
    hessian: HessianOptions = "raw",
    damping: Optional[float] = None,
    profile: bool = False,
    device: str = 'cpu',
    sparsifier_kwargs: Optional[Dict] = None,
    projector_kwargs: Optional[Dict] = None,
    offload: OffloadOptions = "none",
    cache_dir: Optional[str] = None,
) -> IFAttributor:
    """
    Factory function to create an attributor.

    Args:
        attributor_type: Type of attributor to create (currently only "if" is supported)
        setting: Experiment setting/name
        model: PyTorch model
        layer_names: Names of layers to attribute (string or list)
        hessian: Type of Hessian approximation
        damping: Damping factor for Hessian inverse
        profile: Whether to track performance metrics
        device: Device to run computations on
        sparsifier_kwargs: Configuration for sparsifier
        projector_kwargs: Configuration for projector
        offload: Memory management strategy
        cache_dir: Directory for storing data (required for disk offload)

    Returns:
        An attributor instance

    Raises:
        ValueError: If an unknown attributor_type is specified
    """
    if attributor_type.lower() in ["if", "influence", "influence_function", "ifvp"]:
        return IFAttributor(
            setting=setting,
            model=model,
            layer_names=layer_names,
            hessian=hessian,
            damping=damping,
            profile=profile,
            device=device,
            sparsifier_kwargs=sparsifier_kwargs,
            projector_kwargs=projector_kwargs,
            offload=offload,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unknown attributor type: {attributor_type}")