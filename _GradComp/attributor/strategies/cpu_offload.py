from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from .offload_strategy import OffloadStrategy

class CPUOffloadStrategy(OffloadStrategy):
    """
    Strategy that stores data on CPU and moves to device when needed.
    This reduces GPU memory usage by keeping most data on CPU.
    """

    def __init__(self, device: str, layer_names: List[str], cache_dir: Optional[str] = None):
        """
        Initialize the CPU offload strategy.

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
        Store gradients for a batch on CPU.

        Args:
            batch_idx: Batch index
            gradients: List of gradient tensors (one per layer)
            is_test: Whether these are test gradients
        """
        # Move to CPU
        cpu_gradients = [grad.cpu() if grad.device.type != 'cpu' else grad for grad in gradients]

        if is_test:
            self.cached_test_gradients[batch_idx] = cpu_gradients
        else:
            self.cached_gradients[batch_idx] = cpu_gradients

    def retrieve_gradients(self, batch_idx: int, is_test: bool = False) -> List[torch.Tensor]:
        """
        Retrieve gradients for a batch and move to device.

        Args:
            batch_idx: Batch index
            is_test: Whether to retrieve test gradients

        Returns:
            List of gradient tensors (one per layer) on the compute device
        """
        cached_dict = self.cached_test_gradients if is_test else self.cached_gradients

        if batch_idx not in cached_dict:
            return [torch.tensor([], device=self.device) for _ in self.layer_names]

        # Move to device
        return [grad.to(self.device) if grad.numel() > 0 else grad for grad in cached_dict[batch_idx]]

    def store_preconditioner(self, layer_idx: int, preconditioner: torch.Tensor) -> None:
        """
        Store a preconditioner for a layer on CPU.

        Args:
            layer_idx: Layer index
            preconditioner: Preconditioner tensor
        """
        self.preconditioners[layer_idx] = preconditioner.cpu() if preconditioner.device.type != 'cpu' else preconditioner

    def retrieve_preconditioner(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Retrieve a preconditioner for a layer and move to device.

        Args:
            layer_idx: Layer index

        Returns:
            Preconditioner tensor on the compute device, or None if not found
        """
        if layer_idx >= len(self.preconditioners) or self.preconditioners[layer_idx] is None:
            return None

        return self.preconditioners[layer_idx].to(self.device)

    def store_ifvp(self, batch_idx: int, ifvp: List[torch.Tensor]) -> None:
        """
        Store IFVP for a batch on CPU.

        Args:
            batch_idx: Batch index
            ifvp: List of IFVP tensors (one per layer)
        """
        # Move to CPU
        cpu_ifvp = [ivp.cpu() if ivp.device.type != 'cpu' else ivp for ivp in ifvp]
        self.cached_ifvp[batch_idx] = cpu_ifvp

    def retrieve_ifvp(self, batch_idx: int) -> List[torch.Tensor]:
        """
        Retrieve IFVP for a batch and move to device.

        Args:
            batch_idx: Batch index

        Returns:
            List of IFVP tensors (one per layer) on the compute device
        """
        if batch_idx not in self.cached_ifvp:
            return [torch.tensor([], device=self.device) for _ in self.layer_names]

        # Move to device
        return [ivp.to(self.device) if ivp.numel() > 0 else ivp for ivp in self.cached_ifvp[batch_idx]]

    def create_gradient_dataloader(self, data_type: str, batch_size: int = 1,
                                pin_memory: bool = True, batch_range: Optional[Tuple[int, int]] = None,
                                is_test: bool = False) -> Optional[DataLoader]:
        """
        No DataLoader needed for CPU offload, as we access memory directly.

        Returns:
            None
        """
        return None

    def has_preconditioners(self) -> bool:
        """
        Check if preconditioners are available.

        Returns:
            True if any preconditioners are available, False otherwise
        """
        return any(p is not None for p in self.preconditioners)

    def has_ifvp(self) -> bool:
        """
        Check if IFVP are available.

        Returns:
            True if any IFVP are available, False otherwise
        """
        return len(self.cached_ifvp) > 0

    def clear_cache(self) -> None:
        """
        Clear all cached data.
        """
        self.cached_gradients = {}
        self.cached_test_gradients = {}
        self.preconditioners = [None] * len(self.layer_names)
        self.cached_ifvp = {}

    def wait_for_async_operations(self) -> None:
        """
        No asynchronous operations for CPU offload.
        """
        pass

    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Move a tensor to the compute device.

        Args:
            tensor: Input tensor (on CPU)

        Returns:
            Tensor on the compute device
        """
        return tensor.to(self.device) if tensor.device.type == 'cpu' else tensor

    def move_from_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Move a tensor from the compute device to CPU.

        Args:
            tensor: Input tensor on compute device

        Returns:
            Tensor on CPU
        """
        return tensor.cpu() if tensor.device.type != 'cpu' else tensor