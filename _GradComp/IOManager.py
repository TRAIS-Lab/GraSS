# from __future__ import annotations

# from typing import TYPE_CHECKING, List, Dict, Optional, Literal

# import os
# from concurrent.futures import ThreadPoolExecutor
# import glob
# import contextlib
# import torch

# DataTypeOptions = Literal["gradients", "preconditioners", "ifvp"]
# HessianOptions = Literal["none", "raw", "kfac", "ekfac"]

# @contextlib.contextmanager
# def async_stream():
#     """
#     Context manager for asynchronous CUDA stream operations.
#     All operations within this context will be executed asynchronously.
#     """
#     if torch.cuda.is_available():
#         stream = torch.cuda.Stream()
#         with torch.cuda.stream(stream):
#             yield stream
#         # Ensure stream operations complete before returning
#         stream.synchronize()
#     else:
#         yield None

# class DiskIOManager:
#     """
#     Manager for disk I/O operations with thread pool for parallel processing.
#     Handles standardized paths, async reading/writing, and metadata management.

#     File organization:
#     cache_dir/
#       ├── grad/
#       │   ├── batch_0.pt  # Contains all layer gradients for batch 0
#       │   ├── batch_1.pt
#       │   └── ...
#       ├── ifvp/
#       │   ├── batch_0.pt  # Contains all layer IFVP for batch 0
#       │   ├── batch_1.pt
#       │   └── ...
#       └── precond/
#           ├── layer_0.pt  # One preconditioner per layer
#           ├── layer_1.pt
#           └── ...
#     """

#     def __init__(self, cache_dir: str, setting: str, num_threads: int = 32, hessian: HessianOptions = "raw"):
#         """
#         Initialize the DiskIOManager.

#         Args:
#             cache_dir: Directory to save files
#             setting: Experiment setting/name
#             num_threads: Number of worker threads for I/O operations
#             hessian: Hessian approximation type for path generation
#         """
#         self.cache_dir = cache_dir
#         self.setting = setting
#         self.num_threads = num_threads
#         self.hessian = hessian

#         # Create cache directory if it doesn't exist
#         if cache_dir:
#             os.makedirs(cache_dir, exist_ok=True)

#         # Thread pool for async operations
#         self.executor = ThreadPoolExecutor(max_workers=num_threads)
#         self.futures = []

#     def get_path(self,
#                  data_type: DataTypeOptions,
#                  batch_idx: Optional[int] = None,
#                  layer_idx: Optional[int] = None,
#                  is_test: bool = False) -> str:
#         """
#         Generate standardized path for data storage.

#         Args:
#             data_type: Type of data ('gradients', 'preconditioners', or 'ifvp')
#             batch_idx: Optional batch index for batch-specific files
#             layer_idx: Optional layer index for preconditioners
#             is_test: Whether this is for test data

#         Returns:
#             Full path to the file
#         """
#         if not self.cache_dir:
#             raise ValueError("Cache directory is not set")

#         # Determine subdirectory based on data type
#         if data_type == 'gradients':
#             subdir = 'grad'
#         elif data_type == 'preconditioners':
#             subdir = 'precond'
#         elif data_type == 'ifvp':
#             subdir = 'ifvp'
#         else:
#             raise ValueError(f"Unknown data type: {data_type}")

#         # Create subdirectory if it doesn't exist
#         subdir_path = os.path.join(self.cache_dir, subdir)
#         os.makedirs(subdir_path, exist_ok=True)

#         # Determine filename
#         if data_type == 'preconditioners':
#             if layer_idx is None:
#                 raise ValueError("Layer index must be provided for preconditioners")
#             filename = f"layer_{layer_idx}.pt"
#         else:
#             if batch_idx is None:
#                 raise ValueError("Batch index must be provided for gradients and IFVP")
#             prefix = "test_" if is_test else ""
#             filename = f"{prefix}batch_{batch_idx}.pt"

#         return os.path.join(subdir_path, filename)

#     def save_tensor(self, tensor: torch.Tensor, path: str, async_save: bool = True) -> None:
#         """
#         Save a tensor to disk, optionally asynchronously.
#         Uses CUDA stream for efficient CPU transfer and thread pool for disk I/O.

#         Args:
#             tensor: The tensor to save
#             path: Path where to save the tensor
#             async_save: Whether to save asynchronously
#         """
#         # Ensure the directory exists
#         os.makedirs(os.path.dirname(path), exist_ok=True)

#         # Use async stream for CPU transfer if tensor is on CUDA
#         if tensor.is_cuda:
#             with async_stream():
#                 cpu_tensor = tensor.cpu()

#                 if async_save:
#                     # Save asynchronously using thread pool
#                     future = self.executor.submit(torch.save, cpu_tensor, path)
#                     self.futures.append(future)
#                 else:
#                     # Save synchronously
#                     torch.save(cpu_tensor, path)
#         else:
#             # Tensor already on CPU
#             if async_save:
#                 future = self.executor.submit(torch.save, tensor, path)
#                 self.futures.append(future)
#             else:
#                 torch.save(tensor, path)

#     def save_dict(self, data_dict: Dict, path: str, async_save: bool = True) -> None:
#         """
#         Save a dictionary of tensors to disk, optionally asynchronously.
#         Uses CUDA stream for efficient CPU transfer and thread pool for disk I/O.

#         Args:
#             data_dict: Dictionary of tensors to save
#             path: Path where to save the dictionary
#             async_save: Whether to save asynchronously
#         """
#         # Ensure the directory exists
#         os.makedirs(os.path.dirname(path), exist_ok=True)

#         # Process CUDA tensors if present
#         has_cuda = any(isinstance(v, torch.Tensor) and v.is_cuda for v in data_dict.values())

#         if has_cuda:
#             with async_stream():
#                 # Transfer all tensors to CPU within the stream
#                 cpu_dict = {k: v.cpu() if isinstance(v, torch.Tensor) and v.is_cuda else v
#                           for k, v in data_dict.items()}

#                 if async_save:
#                     # Save asynchronously using thread pool
#                     future = self.executor.submit(torch.save, cpu_dict, path)
#                     self.futures.append(future)
#                 else:
#                     # Save synchronously
#                     torch.save(cpu_dict, path)
#         else:
#             # No CUDA tensors
#             if async_save:
#                 future = self.executor.submit(torch.save, data_dict, path)
#                 self.futures.append(future)
#             else:
#                 torch.save(data_dict, path)

#     def load_tensor(self, path: str) -> torch.Tensor:
#         """
#         Load a tensor from disk.

#         Args:
#             path: Path to the tensor file

#         Returns:
#             The loaded tensor
#         """
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Could not find tensor file: {path}")

#         return torch.load(path)

#     def load_dict(self, path: str) -> Dict:
#         """
#         Load a dictionary of tensors from disk.

#         Args:
#             path: Path to the dictionary file

#         Returns:
#             The loaded dictionary
#         """
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Could not find dictionary file: {path}")

#         return torch.load(path)

#     def batch_load_tensors(self, paths: List[str], process_fn=None) -> List[torch.Tensor]:
#         """
#         Load multiple tensors in parallel.

#         Args:
#             paths: List of paths to tensor files
#             process_fn: Optional function to process each tensor after loading

#         Returns:
#             List of loaded tensors
#         """
#         # Submit all load jobs to thread pool
#         futures = [self.executor.submit(self.load_tensor, path) for path in paths]

#         # Wait for all to complete and gather results
#         tensors = []
#         for future in futures:
#             tensor = future.result()
#             if process_fn:
#                 tensor = process_fn(tensor)
#             tensors.append(tensor)

#         return tensors

#     def batch_load_dicts(self, paths: List[str]) -> List[Dict]:
#         """
#         Load multiple dictionaries in parallel.

#         Args:
#             paths: List of paths to dictionary files

#         Returns:
#             List of loaded dictionaries
#         """
#         # Submit all load jobs to thread pool
#         futures = [self.executor.submit(self.load_dict, path) for path in paths]

#         # Wait for all to complete and gather results
#         return [future.result() for future in futures]

#     def find_batch_files(self,
#                        data_type: DataTypeOptions,
#                        is_test: bool = False) -> List[str]:
#         """
#         Find all batch files for a specific data type.

#         Args:
#             data_type: Type of data
#             is_test: Whether to look for test data

#         Returns:
#             List of file paths
#         """
#         if not self.cache_dir:
#             return []

#         # Determine subdirectory based on data type
#         if data_type == 'gradients':
#             subdir = 'grad'
#         elif data_type == 'ifvp':
#             subdir = 'ifvp'
#         else:
#             raise ValueError(f"Cannot find batch files for data type: {data_type}")

#         # Construct path pattern
#         prefix = "test_" if is_test else ""
#         pattern = os.path.join(self.cache_dir, subdir, f"{prefix}batch_*.pt")

#         # Find all matching files
#         return sorted(glob.glob(pattern))

#     def extract_batch_idx(self, path: str) -> int:
#         """
#         Extract batch index from file path.

#         Args:
#             path: File path

#         Returns:
#             Batch index
#         """
#         filename = os.path.basename(path)
#         # Extract batch index from filename (e.g., "batch_42.pt" -> 42)
#         try:
#             return int(filename.split('_')[1].split('.')[0])
#         except (IndexError, ValueError):
#             raise ValueError(f"Could not extract batch index from {filename}")

#     def wait_for_async_operations(self) -> None:
#         """Wait for all pending async operations to complete."""
#         for future in self.futures:
#             future.result()
#         self.futures = []

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Optional, Literal, Tuple, Union
import os
from concurrent.futures import ThreadPoolExecutor, Future
import glob
import contextlib
import torch
from tqdm import tqdm
import time
from threading import Lock
import functools
import queue

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

class LRUCache:
    """
    Simple LRU cache implementation for file data.
    Tracks both size and access time to manage memory.
    """
    def __init__(self, max_size_gb=2):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
        self.current_size = 0
        self.lock = Lock()

    def get(self, key):
        """Get an item from the cache"""
        with self.lock:
            if key in self.cache:
                # Move to the end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None

    def put(self, key, value):
        """Add an item to the cache"""
        with self.lock:
            # Approximate size in bytes (rough estimate for PyTorch tensors)
            if isinstance(value, torch.Tensor):
                size = value.element_size() * value.nelement()
            elif isinstance(value, dict) and all(isinstance(v, torch.Tensor) for v in value.values()):
                size = sum(v.element_size() * v.nelement() for v in value.values())
            else:
                # For other objects, use a default size of 1MB
                size = 1024 * 1024

            # If this item alone is larger than cache, don't cache it
            if size > self.max_size:
                return

            # Remove items if necessary to make space
            while self.current_size + size > self.max_size and self.access_order:
                oldest_key = self.access_order[0]
                oldest_item = self.cache[oldest_key]

                if isinstance(oldest_item, torch.Tensor):
                    oldest_size = oldest_item.element_size() * oldest_item.nelement()
                elif isinstance(oldest_item, dict) and all(isinstance(v, torch.Tensor) for v in oldest_item.values()):
                    oldest_size = sum(v.element_size() * v.nelement() for v in oldest_item.values())
                else:
                    oldest_size = 1024 * 1024

                del self.cache[oldest_key]
                self.access_order.pop(0)
                self.current_size -= oldest_size

            # Add the new item
            if key in self.cache:
                # Update existing entry
                self.access_order.remove(key)
                old_value = self.cache[key]
                if isinstance(old_value, torch.Tensor):
                    old_size = old_value.element_size() * old_value.nelement()
                elif isinstance(old_value, dict) and all(isinstance(v, torch.Tensor) for v in old_value.values()):
                    old_size = sum(v.element_size() * v.nelement() for v in old_value.values())
                else:
                    old_size = 1024 * 1024
                self.current_size -= old_size

            self.cache[key] = value
            self.access_order.append(key)
            self.current_size += size

    def clear(self):
        """Clear the cache"""
        with self.lock:
            self.cache = {}
            self.access_order = []
            self.current_size = 0

class DiskIOManager:
    """
    Enhanced Manager for disk I/O operations with prefetching, caching, and optimized parallel processing.
    Handles standardized paths, async reading/writing, and metadata management.
    """

    def __init__(self, cache_dir: str, setting: str, num_threads: int = 32, hessian: HessianOptions = "raw", cache_size_gb: float = 2.0):
        """
        Initialize the DiskIOManager.

        Args:
            cache_dir: Directory to save files
            setting: Experiment setting/name
            num_threads: Number of worker threads for I/O operations
            hessian: Hessian approximation type for path generation
            cache_size_gb: Maximum size for in-memory file cache in GB
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

        # LRU cache for file data
        self.file_cache = LRUCache(max_size_gb=cache_size_gb)

        # Prefetch queue and lock
        self.prefetch_queue = queue.Queue()
        self.prefetch_lock = Lock()
        self.prefetch_futures = []

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
        Load a tensor from disk with caching.

        Args:
            path: Path to the tensor file

        Returns:
            The loaded tensor
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find tensor file: {path}")

        # Check cache first
        cached_tensor = self.file_cache.get(path)
        if cached_tensor is not None:
            return cached_tensor

        # Load from disk
        tensor = torch.load(path)

        # Cache the result
        self.file_cache.put(path, tensor)

        return tensor

    def load_dict(self, path: str) -> Dict:
        """
        Load a dictionary of tensors from disk with caching.

        Args:
            path: Path to the dictionary file

        Returns:
            The loaded dictionary
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find dictionary file: {path}")

        # Check cache first
        cached_dict = self.file_cache.get(path)
        if cached_dict is not None:
            return cached_dict

        # Load from disk
        data_dict = torch.load(path)

        # Cache the result
        self.file_cache.put(path, data_dict)

        return data_dict

    def prefetch_files(self, paths: List[str]) -> None:
        """
        Start asynchronous prefetching of files.

        Args:
            paths: List of file paths to prefetch
        """
        with self.prefetch_lock:
            # Clear any existing prefetch tasks
            for path in paths:
                if self.file_cache.get(path) is None:  # Only prefetch if not already in cache
                    future = self.executor.submit(self._prefetch_file, path)
                    self.prefetch_futures.append(future)

    def _prefetch_file(self, path: str) -> None:
        """
        Helper function to load a file and store it in the cache.

        Args:
            path: Path to the file
        """
        try:
            if path.endswith('.pt'):
                data = torch.load(path)
                self.file_cache.put(path, data)
        except Exception as e:
            print(f"Error prefetching {path}: {e}")

    def wait_for_prefetch(self) -> None:
        """Wait for all prefetch operations to complete."""
        with self.prefetch_lock:
            for future in self.prefetch_futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Prefetch error: {e}")
            self.prefetch_futures = []

    def batch_load_tensors(self, paths: List[str], process_fn=None) -> List[torch.Tensor]:
        """
        Load multiple tensors in parallel with caching.

        Args:
            paths: List of paths to tensor files
            process_fn: Optional function to process each tensor after loading

        Returns:
            List of loaded tensors
        """
        results = []
        missing_paths = []

        # First check cache
        for path in paths:
            cached_data = self.file_cache.get(path)
            if cached_data is not None:
                if process_fn:
                    results.append(process_fn(cached_data))
                else:
                    results.append(cached_data)
            else:
                missing_paths.append(path)
                results.append(None)  # Placeholder

        if not missing_paths:
            return results

        # Submit load jobs for missing files
        futures = {path: self.executor.submit(self._cached_load_tensor, path, process_fn)
                  for path in missing_paths}

        # Fill in results from futures
        for i, path in enumerate(paths):
            if path in futures:
                try:
                    results[i] = futures[path].result()
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    # Keep the placeholder

        return results

    def _cached_load_tensor(self, path: str, process_fn=None) -> torch.Tensor:
        """Helper function to load tensor and cache it"""
        tensor = torch.load(path)
        self.file_cache.put(path, tensor)
        if process_fn:
            return process_fn(tensor)
        return tensor

    def batch_load_dicts(self, paths: List[str]) -> List[Dict]:
        """
        Load multiple dictionaries in parallel with caching.

        Args:
            paths: List of paths to dictionary files

        Returns:
            List of loaded dictionaries
        """
        results = []
        missing_paths = []

        # First check cache
        for path in paths:
            cached_data = self.file_cache.get(path)
            if cached_data is not None:
                results.append(cached_data)
            else:
                missing_paths.append(path)
                results.append(None)  # Placeholder

        if not missing_paths:
            return results

        # Submit load jobs for missing files
        futures = {path: self.executor.submit(self._cached_load_dict, path)
                  for path in missing_paths}

        # Fill in results from futures
        for i, path in enumerate(paths):
            if path in futures:
                try:
                    results[i] = futures[path].result()
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    # Keep the placeholder

        return [r for r in results if r is not None]

    def _cached_load_dict(self, path: str) -> Dict:
        """Helper function to load dict and cache it"""
        data_dict = torch.load(path)
        self.file_cache.put(path, data_dict)
        return data_dict

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

        # Also wait for prefetch operations
        self.wait_for_prefetch()