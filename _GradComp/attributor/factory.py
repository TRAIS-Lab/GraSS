"""
Enhanced factory and utilities for creating attributors with chunked I/O support.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union, Literal

if TYPE_CHECKING:
    import torch.nn as nn

import torch
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
    chunk_size: int = 32,
) -> IFAttributor:
    """
    Enhanced factory function to create an attributor with chunked I/O support.

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
        chunk_size: Number of batches to group into chunks (for disk offload)

    Returns:
        An enhanced attributor instance with chunked I/O support

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
            chunk_size=chunk_size,
        )
    else:
        raise ValueError(f"Unknown attributor type: {attributor_type}")

def create_chunked_attributor(
    setting: str,
    model: 'nn.Module',
    layer_names: Union[str, List[str]],
    cache_dir: str,
    chunk_size: int = 32,
    hessian: HessianOptions = "raw",
    damping: Optional[float] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
) -> IFAttributor:
    """
    Convenience function to create a chunked IF attributor with optimal defaults.

    Args:
        setting: Experiment setting/name
        model: PyTorch model
        layer_names: Names of layers to attribute (string or list)
        cache_dir: Directory for storing chunked data
        chunk_size: Number of batches to group into chunks
        hessian: Type of Hessian approximation
        damping: Damping factor for Hessian inverse
        device: Device to run computations on
        **kwargs: Additional arguments passed to create_attributor

    Returns:
        An enhanced attributor configured for optimal chunked I/O performance
    """
    # Set optimal defaults for chunked processing
    defaults = {
        'offload': 'disk',
        'profile': True,
    }

    # Update with user-provided kwargs
    defaults.update(kwargs)

    return create_attributor(
        attributor_type="if",
        setting=setting,
        model=model,
        layer_names=layer_names,
        hessian=hessian,
        damping=damping,
        device=device,
        cache_dir=cache_dir,
        chunk_size=chunk_size,
        **defaults
    )

def get_recommended_chunk_size(
    model_size_gb: float,
    available_memory_gb: float,
    num_layers: int,
    device: str = 'cuda'
) -> int:
    """
    Get recommended chunk size based on model size and available memory.

    Args:
        model_size_gb: Model size in GB
        available_memory_gb: Available memory in GB
        num_layers: Number of layers being processed
        device: Target device

    Returns:
        Recommended chunk size
    """
    # Simple heuristic based on memory constraints
    if device == 'cpu':
        base_chunk_size = 64
    else:
        base_chunk_size = 32

    # Adjust based on model size
    if model_size_gb < 1.0:
        multiplier = 2.0
    elif model_size_gb < 5.0:
        multiplier = 1.5
    elif model_size_gb < 10.0:
        multiplier = 1.0
    else:
        multiplier = 0.5

    # Adjust based on number of layers
    layer_factor = max(0.5, min(2.0, 32 / max(1, num_layers)))

    # Adjust based on available memory
    memory_factor = min(2.0, available_memory_gb / max(1, model_size_gb))

    recommended_size = int(base_chunk_size * multiplier * layer_factor * memory_factor)

    # Clamp to reasonable bounds
    return max(4, min(128, recommended_size))

def estimate_memory_requirements(
    model: 'nn.Module',
    chunk_size: int,
    batch_size: int = 32,
    sequence_length: int = 512
) -> Dict[str, float]:
    """
    Estimate memory requirements for chunked processing.

    Args:
        model: PyTorch model
        chunk_size: Number of batches per chunk
        batch_size: Batch size for training
        sequence_length: Sequence length (for NLP models)

    Returns:
        Dictionary with memory estimates in GB
    """
    # Calculate model parameters
    model_params = sum(p.numel() for p in model.parameters())
    model_size_gb = model_params * 4 / (1024**3)  # Assume float32

    # Estimate gradient memory
    gradient_size_gb = model_size_gb * chunk_size * batch_size

    # Estimate intermediate activation memory (rough estimate)
    activation_size_gb = sequence_length * batch_size * 1024 * 4 / (1024**3)  # Rough estimate

    # Estimate total memory usage
    total_memory_gb = model_size_gb + gradient_size_gb + activation_size_gb

    return {
        'model_size_gb': model_size_gb,
        'gradient_memory_gb': gradient_size_gb,
        'activation_memory_gb': activation_size_gb,
        'total_estimated_gb': total_memory_gb,
        'recommended_chunk_size': get_recommended_chunk_size(
            model_size_gb,
            torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 16,
            len(list(model.modules()))
        )
    }