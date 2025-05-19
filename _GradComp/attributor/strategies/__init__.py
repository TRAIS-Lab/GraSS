from typing import List, Optional, Literal

from .offload_strategy import OffloadStrategy
from .memory_offload import MemoryOffloadStrategy
from .cpu_offload import CPUOffloadStrategy
from .disk_offload import DiskOffloadStrategy

OffloadOptions = Literal["none", "cpu", "disk"]

def create_offload_strategy(
    offload_type: OffloadOptions,
    device: str,
    layer_names: List[str],
    cache_dir: Optional[str] = None
) -> OffloadStrategy:
    """
    Factory function to create an appropriate offload strategy.

    Args:
        offload_type: Type of offloading to use
        device: Compute device
        layer_names: Names of layers being analyzed
        cache_dir: Directory for caching data (required for disk offload)

    Returns:
        An offload strategy instance

    Raises:
        ValueError: If disk offload is requested but no cache_dir is provided
    """
    if offload_type == "none":
        return MemoryOffloadStrategy(device, layer_names)
    elif offload_type == "cpu":
        return CPUOffloadStrategy(device, layer_names)
    elif offload_type == "disk":
        if cache_dir is None:
            raise ValueError("Cache directory must be provided when using disk offload")
        return DiskOffloadStrategy(device, layer_names, cache_dir)
    else:
        raise ValueError(f"Unknown offload type: {offload_type}")