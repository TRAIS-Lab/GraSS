from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Literal, List, Optional, Union, Tuple, TypedDict, cast
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import glob
import contextlib

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    import torch.nn as nn

import torch
from tqdm import tqdm

class DiskIOManager:
    """
    Manager for disk I/O operations with thread pool for parallel processing.
    Handles standardized paths, async reading/writing, and metadata management.

    File organization:
    cache_dir/
      ├── grad/
      │   ├── batch_0.pt  # Contains all layer gradients for batch 0
      │   ├── batch_1.pt
      │   └── ...
      ├── ifvp/
      │   ├── batch_0.pt  # Contains all layer IFVP for batch 0
      │   ├── batch_1.pt
      │   └── ...
      └── precond/
          ├── layer_0.pt  # One preconditioner per layer
          ├── layer_1.pt
          └── ...
    """

    def __init__(self, cache_dir: str, setting: str, num_threads: int = 32, hessian: HessianOptions = "raw"):
        """
        Initialize the DiskIOManager.

        Args:
            cache_dir: Directory to save files
            setting: Experiment setting/name
            num_threads: Number of worker threads for I/O operations
            hessian: Hessian approximation type for path generation
        """
        self.cache_dir = cache_dir
        self.setting = setting
        self.num_threads = num_threads
        self.hessian = hessian

        # Create cache directory if it doesn't exist
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.futures = []

    def get_path(self,
                 data_type: DataTypeOptions,
                 batch_idx: Optional[int] = None,
                 layer_idx: Optional[int] = None,
                 is_test: bool = False) -> str:
        """
        Generate standardized path for data storage.

        Args:
            data_type: Type of data ('gradients', 'preconditioners', or 'ifvp')
            batch_idx: Optional batch index for batch-specific files
            layer_idx: Optional layer index for preconditioners
            is_test: Whether this is for test data

        Returns:
            Full path to the file
        """
        if not self.cache_dir:
            raise ValueError("Cache directory is not set")

        # Determine subdirectory based on data type
        if data_type == 'gradients':
            subdir = 'grad'
        elif data_type == 'preconditioners':
            subdir = 'precond'
        elif data_type == 'ifvp':
            subdir = 'ifvp'
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        # Create subdirectory if it doesn't exist
        subdir_path = os.path.join(self.cache_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)

        # Determine filename
        if data_type == 'preconditioners':
            if layer_idx is None:
                raise ValueError("Layer index must be provided for preconditioners")
            filename = f"layer_{layer_idx}.pt"
        else:
            if batch_idx is None:
                raise ValueError("Batch index must be provided for gradients and IFVP")
            prefix = "test_" if is_test else ""
            filename = f"{prefix}batch_{batch_idx}.pt"

        return os.path.join(subdir_path, filename)

    def save_tensor(self, tensor: torch.Tensor, path: str, async_save: bool = True) -> None:
        """
        Save a tensor to disk, optionally asynchronously.
        Uses CUDA stream for efficient CPU transfer and thread pool for disk I/O.

        Args:
            tensor: The tensor to save
            path: Path where to save the tensor
            async_save: Whether to save asynchronously
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Use async stream for CPU transfer if tensor is on CUDA
        if tensor.is_cuda:
            with async_stream():
                cpu_tensor = tensor.cpu()

                if async_save:
                    # Save asynchronously using thread pool
                    future = self.executor.submit(torch.save, cpu_tensor, path)
                    self.futures.append(future)
                else:
                    # Save synchronously
                    torch.save(cpu_tensor, path)
        else:
            # Tensor already on CPU
            if async_save:
                future = self.executor.submit(torch.save, tensor, path)
                self.futures.append(future)
            else:
                torch.save(tensor, path)

    def save_dict(self, data_dict: Dict, path: str, async_save: bool = True) -> None:
        """
        Save a dictionary of tensors to disk, optionally asynchronously.
        Uses CUDA stream for efficient CPU transfer and thread pool for disk I/O.

        Args:
            data_dict: Dictionary of tensors to save
            path: Path where to save the dictionary
            async_save: Whether to save asynchronously
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Process CUDA tensors if present
        has_cuda = any(isinstance(v, torch.Tensor) and v.is_cuda for v in data_dict.values())

        if has_cuda:
            with async_stream():
                # Transfer all tensors to CPU within the stream
                cpu_dict = {k: v.cpu() if isinstance(v, torch.Tensor) and v.is_cuda else v
                          for k, v in data_dict.items()}

                if async_save:
                    # Save asynchronously using thread pool
                    future = self.executor.submit(torch.save, cpu_dict, path)
                    self.futures.append(future)
                else:
                    # Save synchronously
                    torch.save(cpu_dict, path)
        else:
            # No CUDA tensors
            if async_save:
                future = self.executor.submit(torch.save, data_dict, path)
                self.futures.append(future)
            else:
                torch.save(data_dict, path)

    def load_tensor(self, path: str) -> torch.Tensor:
        """
        Load a tensor from disk.

        Args:
            path: Path to the tensor file

        Returns:
            The loaded tensor
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find tensor file: {path}")

        return torch.load(path)

    def load_dict(self, path: str) -> Dict:
        """
        Load a dictionary of tensors from disk.

        Args:
            path: Path to the dictionary file

        Returns:
            The loaded dictionary
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find dictionary file: {path}")

        return torch.load(path)

    def batch_load_tensors(self, paths: List[str], process_fn=None) -> List[torch.Tensor]:
        """
        Load multiple tensors in parallel.

        Args:
            paths: List of paths to tensor files
            process_fn: Optional function to process each tensor after loading

        Returns:
            List of loaded tensors
        """
        # Submit all load jobs to thread pool
        futures = [self.executor.submit(self.load_tensor, path) for path in paths]

        # Wait for all to complete and gather results
        tensors = []
        for future in futures:
            tensor = future.result()
            if process_fn:
                tensor = process_fn(tensor)
            tensors.append(tensor)

        return tensors

    def batch_load_dicts(self, paths: List[str]) -> List[Dict]:
        """
        Load multiple dictionaries in parallel.

        Args:
            paths: List of paths to dictionary files

        Returns:
            List of loaded dictionaries
        """
        # Submit all load jobs to thread pool
        futures = [self.executor.submit(self.load_dict, path) for path in paths]

        # Wait for all to complete and gather results
        return [future.result() for future in futures]

    def find_batch_files(self,
                       data_type: DataTypeOptions,
                       is_test: bool = False) -> List[str]:
        """
        Find all batch files for a specific data type.

        Args:
            data_type: Type of data
            is_test: Whether to look for test data

        Returns:
            List of file paths
        """
        if not self.cache_dir:
            return []

        # Determine subdirectory based on data type
        if data_type == 'gradients':
            subdir = 'grad'
        elif data_type == 'ifvp':
            subdir = 'ifvp'
        else:
            raise ValueError(f"Cannot find batch files for data type: {data_type}")

        # Construct path pattern
        prefix = "test_" if is_test else ""
        pattern = os.path.join(self.cache_dir, subdir, f"{prefix}batch_*.pt")

        # Find all matching files
        return sorted(glob.glob(pattern))

    def extract_batch_idx(self, path: str) -> int:
        """
        Extract batch index from file path.

        Args:
            path: File path

        Returns:
            Batch index
        """
        filename = os.path.basename(path)
        # Extract batch index from filename (e.g., "batch_42.pt" -> 42)
        try:
            return int(filename.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            raise ValueError(f"Could not extract batch index from {filename}")

    def wait_for_async_operations(self) -> None:
        """Wait for all pending async operations to complete."""
        for future in self.futures:
            future.result()
        self.futures = []

