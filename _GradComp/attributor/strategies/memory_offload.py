from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from .offload_strategy import OffloadStrategy

class MemoryOffloadStrategy(OffloadStrategy):
    """
    Strategy that keeps all data in memory on the specified device.
    No offloading is performed - data remains on the computation device.
    """

    def __init__(self, device: str, layer_names: List[str], cache_dir: Optional[str] = None):
        """
        Initialize the memory offload strategy.

        Args:
            device: The primary compute device
            layer_names: Names of layers being analyzed
            cache_dir: Directory for caching data (ignored for this strategy)
        """
        self.device = device
        self.layer_names = layer_names
        self.cached_gradients = {}
        self.cached_test_gradients = {}
        self.preconditioners = [None] * len(layer_names)
        self.cached_ifvp = {}

    def store_gradients(self, batch_idx: int, gradients: List[torch.Tensor], is_test: bool = False) -> None:
        """
        Store gradients for a batch in memory on the device.

        Args:
            batch_idx: Batch index
            gradients: List of gradient tensors (one per layer)
            is_test: Whether these are test gradients
        """
        if is_test:
            self.cached_test_gradients[batch_idx] = gradients
        else:
            self.cached_gradients[batch_idx] = gradients

    def retrieve_gradients(self, batch_idx: int, is_test: bool = False) -> List[torch.Tensor]:
        """
        Retrieve gradients for a batch from memory.

        Args:
            batch_idx: Batch index
            is_test: Whether to retrieve test gradients

        Returns:
            List of gradient tensors (one per layer)
        """
        if is_test:
            if batch_idx not in self.cached_test_gradients:
                return [torch.tensor([], device=self.device) for _ in self.layer_names]
            return self.cached_test_gradients[batch_idx]
        else:
            if batch_idx not in self.cached_gradients:
                return [torch.tensor([], device=self.device) for _ in self.layer_names]
            return self.cached_gradients[batch_idx]

    def store_preconditioner(self, layer_idx: int, preconditioner: torch.Tensor) -> None:
        """
        Store a preconditioner for a layer in memory.

        Args:
            layer_idx: Layer index
            preconditioner: Preconditioner tensor
        """
        self.preconditioners[layer_idx] = preconditioner

    def retrieve_preconditioner(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Retrieve a preconditioner for a layer from memory.

        Args:
            layer_idx: Layer index

        Returns:
            Preconditioner tensor, or None if not found
        """
        if layer_idx >= len(self.preconditioners):
            return None
        return self.preconditioners[layer_idx]

    def store_ifvp(self, batch_idx: int, ifvp: List[torch.Tensor]) -> None:
        """
        Store IFVP for a batch in memory.

        Args:
            batch_idx: Batch index
            ifvp: List of IFVP tensors (one per layer)
        """
        self.cached_ifvp[batch_idx] = ifvp

    def retrieve_ifvp(self, batch_idx: int) -> List[torch.Tensor]:
        """
        Retrieve IFVP for a batch from memory.

        Args:
            batch_idx: Batch index

        Returns:
            List of IFVP tensors (one per layer)
        """
        if batch_idx not in self.cached_ifvp:
            return [torch.tensor([], device=self.device) for _ in self.layer_names]
        return self.cached_ifvp[batch_idx]

    def create_gradient_dataloader(self, data_type: str, batch_size: int = 1,
                                num_workers: int = 4, pin_memory: bool = True,
                                batch_range: Optional[Tuple[int, int]] = None,
                                is_test: bool = False) -> Optional[DataLoader]:
        """
        No DataLoader needed for memory offload as we access memory directly.

        Returns:
            None
        """
        return None

    def has_preconditioners(self) -> bool:
        """
        Check if preconditioners are available in memory.

        Returns:
            True if any preconditioners are available, False otherwise
        """
        return any(p is not None for p in self.preconditioners)

    def has_ifvp(self) -> bool:
        """
        Check if IFVP are available in memory.

        Returns:
            True if any IFVP are available, False otherwise
        """
        return len(self.cached_ifvp) > 0

    def clear_cache(self) -> None:
        """
        Clear all cached data from memory.
        """
        self.cached_gradients = {}
        self.cached_test_gradients = {}
        self.preconditioners = [None] * len(self.layer_names)
        self.cached_ifvp = {}

    def wait_for_async_operations(self) -> None:
        """
        No asynchronous operations for memory offload.
        """
        pass

    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        No movement needed as tensors are already on the compute device.

        Args:
            tensor: Input tensor

        Returns:
            Same tensor (already on the device)
        """
        return tensor

    def move_from_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        No movement needed as we keep tensors on the device.

        Args:
            tensor: Input tensor

        Returns:
            Same tensor (stays on the device)
        """
        return tensor