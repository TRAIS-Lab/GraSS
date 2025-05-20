"""
Disk I/O manager for efficient data storage and retrieval.
"""

from typing import TYPE_CHECKING, List, Dict, Optional, Literal, Any, Union, Tuple
import os
from concurrent.futures import ThreadPoolExecutor
import glob
import contextlib
import torch
import numpy as np
import time
import json
import gc
import logging

from .memory_map import MemoryMapHandler

# Configure logger
logger = logging.getLogger(__name__)

# Type definitions
DataTypeOptions = Literal["gradients", "preconditioners", "ifvp"]
HessianOptions = Literal["none", "raw", "kfac", "ekfac"]

@contextlib.contextmanager
def async_stream():
    """
    Context manager for asynchronous CUDA stream operations.
    All operations within this context will be executed asynchronously.
    """
    if torch.cuda.is_available():
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            yield stream
        # Ensure stream operations complete before returning
        stream.synchronize()
    else:
        yield None

class DiskIOManager:
    """
    Manager for disk I/O operations with thread pool for parallel processing.
    Uses memory-mapped files for efficient data access.

    File organization:
    cache_dir/
      ├── grad/
      │   ├── batch_0.mmap  # Raw gradient data
      │   ├── batch_0_metadata.json  # Metadata for the batch
      │   └── ...
      ├── ifvp/
      │   ├── batch_0.mmap
      │   ├── batch_0_metadata.json
      │   └── ...
      └── precond/
          ├── layer_0.pt  # One preconditioner per layer (standard PT files)
          ├── layer_1.pt
          └── ...
    """

    def __init__(self, cache_dir: str, setting: str, num_threads: int = 16, hessian: HessianOptions = "raw"):
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
            for subdir in ['grad', 'ifvp', 'precond']:
                os.makedirs(os.path.join(cache_dir, subdir), exist_ok=True)
            logger.info(f"Created cache directories in {cache_dir}")

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.futures = []

        # Buffer for prefetched data
        self.prefetch_cache = {}

        # Default memory-mapped file data type
        self.default_dtype = 'float32'

        logger.info(f"Initialized DiskIOManager with {num_threads} worker threads")

    def get_path(
            self,
            data_type: DataTypeOptions,
            batch_idx: Optional[int] = None,
            layer_idx: Optional[int] = None,
            is_test: bool = False
        ) -> str:
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
            ext = '.mmap'
        elif data_type == 'preconditioners':
            subdir = 'precond'
            ext = '.pt'
        elif data_type == 'ifvp':
            subdir = 'ifvp'
            ext = '.mmap'
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        # Determine filename
        if data_type == 'preconditioners':
            if layer_idx is None:
                raise ValueError("Layer index must be provided for preconditioners")
            filename = f"layer_{layer_idx}{ext}"
        else:
            if batch_idx is None:
                raise ValueError("Batch index must be provided for gradients and IFVP")
            prefix = "test_" if is_test else ""
            filename = f"{prefix}batch_{batch_idx}{ext}"

        return os.path.join(self.cache_dir, subdir, filename)

    def get_base_path(
            self,
            data_type: DataTypeOptions,
            batch_idx: Optional[int] = None,
            layer_idx: Optional[int] = None,
            is_test: bool = False
        ) -> str:
        """
        Generate base path without extension for data storage.

        Args:
            data_type: Type of data ('gradients', 'preconditioners', or 'ifvp')
            batch_idx: Optional batch index for batch-specific files
            layer_idx: Optional layer index for preconditioners
            is_test: Whether this is for test data

        Returns:
            Base path without extension
        """
        full_path = self.get_path(data_type, batch_idx, layer_idx, is_test)
        return os.path.splitext(full_path)[0]

    def save_tensor(self, tensor: torch.Tensor, path: str, async_save: bool = True) -> None:
        """
        Save a tensor to disk, optionally asynchronously.
        Uses standard PyTorch save for preconditioners.

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

    def save_dict_mmap(
            self,
            data_dict: Dict[int, torch.Tensor],
            path: str,
            batch_idx: int,
            async_save: bool = True
        ) -> None:
        """
        Save a dictionary of tensors to a memory-mapped file.

        Args:
            data_dict: Dictionary mapping layer indices to tensors
            path: Path for the memory-mapped file
            batch_idx: Batch index for metadata
            async_save: Whether to save asynchronously
        """
        # Extract base path and directory
        dir_path = os.path.dirname(path)
        file_name = os.path.basename(path)

        if async_save:
            future = self.executor.submit(
                MemoryMapHandler.write,
                dir_path,
                file_name,
                batch_idx,
                data_dict,
                self.default_dtype
            )
            self.futures.append(future)
        else:
            MemoryMapHandler.write(
                dir_path,
                file_name,
                batch_idx,
                data_dict,
                self.default_dtype
            )

    def save_dict(self, data_dict: Dict[int, torch.Tensor], path: str,
                 batch_idx: Optional[int] = None, async_save: bool = True) -> None:
        """
        Save a dictionary of tensors to disk.
        For gradients and IFVP, uses memory-mapped files.

        Args:
            data_dict: Dictionary of tensors to save
            path: Path where to save the dictionary
            batch_idx: Batch index for memory-mapped file metadata
            async_save: Whether to save asynchronously
        """
        # Check if this is a path for memory-mapped files (gradients or IFVP)
        if path.endswith('.mmap') or ('grad' in path or 'ifvp' in path):
            # Ensure batch_idx is provided for memory-mapped files
            if batch_idx is None:
                batch_idx = self.extract_batch_idx(path)

            # Use memory-mapped file storage
            self.save_dict_mmap(data_dict, path, batch_idx, async_save=async_save)
        else:
            # Use standard PyTorch save for other data (e.g., preconditioners)
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
        Load a tensor from disk. For preconditioners only.

        Args:
            path: Path to the tensor file

        Returns:
            The loaded tensor
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find tensor file: {path}")

        return torch.load(path)

    def load_dict(self, path: str) -> Dict[int, torch.Tensor]:
        """
        Load a dictionary of tensors from disk.
        For gradients and IFVP, uses memory-mapped files.

        Args:
            path: Path to the dictionary file

        Returns:
            The loaded dictionary
        """
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find file: {path}")

        # Check if this is a memory-mapped file
        if path.endswith('.mmap'):
            # Load from memory-mapped file
            dir_path = os.path.dirname(path)
            file_name = os.path.basename(path)
            return MemoryMapHandler.load_tensor_dict(dir_path, file_name)
        else:
            # Standard PyTorch load
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
        # Check if any paths are already in prefetch cache
        results = []
        paths_to_load = []
        indices_to_load = []

        for i, path in enumerate(paths):
            if path in self.prefetch_cache:
                results.append(self.prefetch_cache[path])
                del self.prefetch_cache[path]  # Remove from cache once used
            else:
                paths_to_load.append(path)
                indices_to_load.append(i)

        # If all paths were in cache, return immediately
        if not paths_to_load:
            return results

        # Submit all load jobs to thread pool
        futures = [self.executor.submit(self.load_tensor, path) for path in paths_to_load]

        # Wait for all to complete and gather results
        loaded_tensors = []
        for future in futures:
            tensor = future.result()
            if process_fn:
                tensor = process_fn(tensor)
            loaded_tensors.append(tensor)

        # Merge cached and newly loaded tensors in the correct order
        for idx, tensor in zip(indices_to_load, loaded_tensors):
            results.insert(idx, tensor)

        return results

    def batch_load_dicts(self, paths: List[str]) -> List[Dict[int, torch.Tensor]]:
        """
        Load multiple dictionaries in parallel.

        Args:
            paths: List of paths to dictionary files

        Returns:
            List of loaded dictionaries
        """
        # Check if any paths are already in prefetch cache
        results = []
        paths_to_load = []
        indices_to_load = []

        for i, path in enumerate(paths):
            if path in self.prefetch_cache:
                results.append(self.prefetch_cache[path])
                del self.prefetch_cache[path]  # Remove from cache once used
            else:
                paths_to_load.append(path)
                indices_to_load.append(i)

        # If all paths were in cache, return immediately
        if not paths_to_load:
            return results

        # Submit all load jobs to thread pool
        futures = [self.executor.submit(self.load_dict, path) for path in paths_to_load]

        # Wait for all to complete and gather results
        loaded_dicts = [future.result() for future in futures]

        # Merge cached and newly loaded dicts in the correct order
        for idx, dict_obj in zip(indices_to_load, loaded_dicts):
            results.insert(idx, dict_obj)

        return results

    def prefetch_files(self, paths: List[str], dict_mode: bool = True) -> None:
        """
        Prefetch files into memory to speed up future access.

        Args:
            paths: List of paths to prefetch
            dict_mode: Whether to load as dicts (True) or tensors (False)
        """
        # Skip paths already in cache
        paths = [p for p in paths if p not in self.prefetch_cache]
        if not paths:
            return

        # Submit jobs to thread pool
        load_fn = self.load_dict if dict_mode else self.load_tensor
        futures = [(path, self.executor.submit(load_fn, path)) for path in paths]

        # Store results in cache without blocking
        def _cache_result(future_tuple):
            path, future = future_tuple
            try:
                self.prefetch_cache[path] = future.result()
            except Exception as e:
                logger.error(f"Error prefetching {path}: {e}")

        # Submit cache updates to thread pool
        for future_tuple in futures:
            self.executor.submit(_cache_result, future_tuple)

    def find_batch_files(self, data_type: DataTypeOptions, is_test: bool = False) -> List[str]:
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
        pattern = os.path.join(self.cache_dir, subdir, f"{prefix}batch_*.mmap")

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
        # Extract batch index from filename (e.g., "batch_42.mmap" -> 42)
        try:
            return int(filename.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            raise ValueError(f"Could not extract batch index from {filename}")

    def wait_for_async_operations(self) -> None:
        """Wait for all pending async operations to complete."""
        for future in self.futures:
            future.result()
        self.futures = []
        logger.debug("All async disk operations completed")

    def clear_prefetch_cache(self) -> None:
        """Clear the prefetch cache to free memory."""
        self.prefetch_cache.clear()
        logger.debug("Cleared prefetch cache")