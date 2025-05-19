import os
from typing import Dict, List, Optional, Tuple, Union, Set

import torch
from torch.utils.data import DataLoader

from .offload_strategy import OffloadStrategy
from ...io.disk_io import DiskIOManager
from ...data.dataset import GradientDataset, custom_collate_fn

class DiskOffloadStrategy(OffloadStrategy):
    """
    Strategy that stores data on disk and loads when needed.
    This minimizes memory usage but has higher I/O overhead.
    """

    def __init__(self, device: str, layer_names: List[str], cache_dir: Optional[str] = None):
        """
        Initialize the disk offload strategy.

        Args:
            device: The primary compute device
            layer_names: Names of layers being analyzed
            cache_dir: Directory for caching data (required for this strategy)
        """
        if cache_dir is None:
            raise ValueError("Cache directory must be provided for disk offload")

        self.device = device
        self.layer_names = layer_names
        self.cache_dir = cache_dir
        self.disk_io = DiskIOManager(cache_dir, "default", hessian="raw")

        # Lightweight sets to track available batches
        self.train_batch_indices: Set[int] = set()
        self.test_batch_indices: Set[int] = set()
        self.ifvp_batch_indices: Set[int] = set()
        self.has_stored_preconditioners = False

    def store_gradients(self, batch_idx: int, gradients: List[torch.Tensor], is_test: bool = False) -> None:
        """
        Store gradients for a batch on disk.

        Args:
            batch_idx: Batch index
            gradients: List of gradient tensors (one per layer)
            is_test: Whether these are test gradients
        """
        # Move to CPU first
        cpu_gradients = [grad.cpu() if grad.device.type != 'cpu' else grad for grad in gradients]

        # Create dictionary with layer indices as keys
        grad_dict = {idx: grad for idx, grad in enumerate(cpu_gradients)}

        # Save to disk
        file_path = self.disk_io.get_path(
            data_type='gradients',
            batch_idx=batch_idx,
            is_test=is_test
        )
        self.disk_io.save_dict(grad_dict, file_path, batch_idx=batch_idx)

        # Update tracking
        if is_test:
            self.test_batch_indices.add(batch_idx)
        else:
            self.train_batch_indices.add(batch_idx)

    def retrieve_gradients(self, batch_idx: int, is_test: bool = False) -> List[torch.Tensor]:
        """
        Retrieve gradients for a batch from disk and move to device.

        Args:
            batch_idx: Batch index
            is_test: Whether to retrieve test gradients

        Returns:
            List of gradient tensors (one per layer) on the compute device
        """
        file_path = self.disk_io.get_path(
            data_type='gradients',
            batch_idx=batch_idx,
            is_test=is_test
        )

        if not os.path.exists(file_path):
            return [torch.tensor([], device=self.device) for _ in self.layer_names]

        # Load from disk
        grad_dict = self.disk_io.load_dict(file_path)

        # Convert to list and move to device
        result = []
        for layer_idx in range(len(self.layer_names)):
            if layer_idx in grad_dict and grad_dict[layer_idx].numel() > 0:
                result.append(grad_dict[layer_idx].to(self.device))
            else:
                result.append(torch.tensor([], device=self.device))

        return result

    def store_preconditioner(self, layer_idx: int, preconditioner: torch.Tensor) -> None:
        """
        Store a preconditioner for a layer on disk.

        Args:
            layer_idx: Layer index
            preconditioner: Preconditioner tensor
        """
        # Move to CPU first
        cpu_precond = preconditioner.cpu() if preconditioner.device.type != 'cpu' else preconditioner

        # Save to disk
        file_path = self.disk_io.get_path('preconditioners', layer_idx=layer_idx)
        self.disk_io.save_tensor(cpu_precond, file_path)

        # Update tracking
        self.has_stored_preconditioners = True

    def retrieve_preconditioner(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Retrieve a preconditioner for a layer from disk and move to device.

        Args:
            layer_idx: Layer index

        Returns:
            Preconditioner tensor on the compute device, or None if not found
        """
        file_path = self.disk_io.get_path('preconditioners', layer_idx=layer_idx)

        if not os.path.exists(file_path):
            return None

        # Load from disk and move to device
        return self.disk_io.load_tensor(file_path).to(self.device)

    def store_ifvp(self, batch_idx: int, ifvp: List[torch.Tensor]) -> None:
        """
        Store IFVP for a batch on disk.

        Args:
            batch_idx: Batch index
            ifvp: List of IFVP tensors (one per layer)
        """
        # Move to CPU first
        cpu_ifvp = [ivp.cpu() if ivp.device.type != 'cpu' else ivp for ivp in ifvp]

        # Create dictionary with layer indices as keys
        ifvp_dict = {idx: ivp for idx, ivp in enumerate(cpu_ifvp)}

        # Save to disk
        file_path = self.disk_io.get_path('ifvp', batch_idx=batch_idx)
        self.disk_io.save_dict(ifvp_dict, file_path, batch_idx=batch_idx)

        # Update tracking
        self.ifvp_batch_indices.add(batch_idx)

    def retrieve_ifvp(self, batch_idx: int) -> List[torch.Tensor]:
        """
        Retrieve IFVP for a batch from disk and move to device.

        Args:
            batch_idx: Batch index

        Returns:
            List of IFVP tensors (one per layer) on the compute device
        """
        file_path = self.disk_io.get_path('ifvp', batch_idx=batch_idx)

        if not os.path.exists(file_path):
            return [torch.tensor([], device=self.device) for _ in self.layer_names]

        # Load from disk
        ifvp_dict = self.disk_io.load_dict(file_path)

        # Convert to list and move to device
        result = []
        for layer_idx in range(len(self.layer_names)):
            if layer_idx in ifvp_dict and ifvp_dict[layer_idx].numel() > 0:
                result.append(ifvp_dict[layer_idx].to(self.device))
            else:
                result.append(torch.tensor([], device=self.device))

        return result

    def create_gradient_dataloader(self, data_type: str, batch_size: int = 1,
                                num_workers: int = 4, pin_memory: bool = True,
                                batch_range: Optional[Tuple[int, int]] = None,
                                is_test: bool = False) -> DataLoader:
        """
        Create a DataLoader for loading data from disk.

        Args:
            data_type: Type of data to load ("gradients" or "ifvp")
            batch_size: Batch size
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            batch_range: Optional range of batches to include
            is_test: Whether to load test data

        Returns:
            DataLoader for efficient loading of data files
        """
        dataset = GradientDataset(self.disk_io, data_type, batch_range, is_test, self.layer_names)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=custom_collate_fn
        )

    def has_preconditioners(self) -> bool:
        """
        Check if preconditioners are available on disk.

        Returns:
            True if any preconditioners are available, False otherwise
        """
        if self.has_stored_preconditioners:
            return True

        # Check if any preconditioner file exists
        for layer_idx in range(len(self.layer_names)):
            file_path = self.disk_io.get_path('preconditioners', layer_idx=layer_idx)
            if os.path.exists(file_path):
                self.has_stored_preconditioners = True
                return True

        return False

    def has_ifvp(self) -> bool:
        """
        Check if IFVP are available on disk.

        Returns:
            True if any IFVP are available, False otherwise
        """
        if self.ifvp_batch_indices:
            return True

        # Find all IFVP files
        ifvp_files = self.disk_io.find_batch_files('ifvp')
        if ifvp_files:
            # Update tracking
            for file_path in ifvp_files:
                batch_idx = self.disk_io.extract_batch_idx(file_path)
                self.ifvp_batch_indices.add(batch_idx)
            return True

        return False

    def clear_cache(self) -> None:
        """
        Clear all cached data from disk.
        """
        # Remove gradient files
        for batch_idx in self.train_batch_indices:
            file_path = self.disk_io.get_path('gradients', batch_idx=batch_idx)
            if os.path.exists(file_path):
                os.remove(file_path)

        for batch_idx in self.test_batch_indices:
            file_path = self.disk_io.get_path('gradients', batch_idx=batch_idx, is_test=True)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Remove preconditioner files
        for layer_idx in range(len(self.layer_names)):
            file_path = self.disk_io.get_path('preconditioners', layer_idx=layer_idx)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Remove IFVP files
        for batch_idx in self.ifvp_batch_indices:
            file_path = self.disk_io.get_path('ifvp', batch_idx=batch_idx)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Reset tracking
        self.train_batch_indices = set()
        self.test_batch_indices = set()
        self.has_stored_preconditioners = False
        self.ifvp_batch_indices = set()

    def wait_for_async_operations(self) -> None:
        """
        Wait for any pending asynchronous disk operations to complete.
        """
        self.disk_io.wait_for_async_operations()

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
        Move a tensor from the compute device to CPU for disk storage.

        Args:
            tensor: Input tensor on compute device

        Returns:
            Tensor on CPU
        """
        return tensor.cpu() if tensor.device.type != 'cpu' else tensor