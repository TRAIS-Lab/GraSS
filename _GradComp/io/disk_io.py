"""
Enhanced Disk I/O manager with chunking support for efficient data storage and retrieval.
"""

import os
import gc
import logging
import time
import threading
from typing import List, Dict, Optional, Literal, Any, Union, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np

from .memory_map import ChunkedMemoryMapHandler

logger = logging.getLogger(__name__)

DataTypeOptions = Literal["gradients", "preconditioners", "ifvp"]
HessianOptions = Literal["none", "raw", "kfac", "ekfac"]

def align_batch_ranges_to_chunks(total_batches: int, chunk_size: int, num_workers: int) -> List[Tuple[int, int]]:
    """
    Create batch ranges aligned to chunk boundaries.

    Args:
        total_batches: Total number of batches
        chunk_size: Size of each chunk
        num_workers: Number of workers

    Returns:
        List of (start, end) batch ranges aligned to chunk boundaries
    """
    total_chunks = (total_batches + chunk_size - 1) // chunk_size
    chunks_per_worker = (total_chunks + num_workers - 1) // num_workers

    batch_ranges = []
    for worker_id in range(num_workers):
        start_chunk = worker_id * chunks_per_worker
        end_chunk = min((worker_id + 1) * chunks_per_worker, total_chunks)

        if start_chunk >= total_chunks:
            break

        start_batch = start_chunk * chunk_size
        end_batch = min(end_chunk * chunk_size, total_batches)

        if start_batch < total_batches:
            batch_ranges.append((start_batch, end_batch))

    return batch_ranges

class ChunkedDiskIOManager:
    """
    Enhanced manager for disk I/O operations with immediate flushing.
    Uses memory-mapped files with immediate writing to prevent data loss.
    """

    def __init__(self, cache_dir: str, setting: str, num_threads: int = 16,
                 hessian: HessianOptions = "raw", chunk_size: int = 32):
        self.cache_dir = cache_dir
        self.setting = setting
        self.num_threads = num_threads
        self.hessian = hessian
        self.chunk_size = chunk_size

        # Create cache directory if it doesn't exist
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            for subdir in ['grad', 'ifvp', 'precond']:
                os.makedirs(os.path.join(cache_dir, subdir), exist_ok=True)

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.futures = []

        # Immediate chunk storage: accumulate batches and flush when chunk is complete
        self._immediate_chunk_storage = defaultdict(dict)  # data_type -> chunk_id -> batch_data
        self._storage_locks = defaultdict(threading.Lock)

        # Track current batch range being processed
        self.current_batch_range = None

        logger.info(f"Initialized ChunkedDiskIOManager with immediate flushing, chunk_size={chunk_size}")

    def get_chunk_id(self, batch_idx: int) -> int:
        """Get chunk ID for a batch index."""
        return batch_idx // self.chunk_size

    def get_chunk_start_batch(self, chunk_id: int) -> int:
        """Get the starting batch index for a chunk."""
        return chunk_id * self.chunk_size

    def get_chunk_end_batch(self, chunk_id: int) -> int:
        """Get the ending batch index for a chunk."""
        return (chunk_id + 1) * self.chunk_size

    def start_batch_range(self, start_batch: int, end_batch: int):
        """Start processing a batch range. Validates chunk alignment."""
        if start_batch % self.chunk_size != 0:
            raise ValueError(f"Batch range start {start_batch} must be aligned to chunk_size {self.chunk_size}")

        self.current_batch_range = (start_batch, end_batch)
        logger.info(f"Starting batch range [{start_batch}, {end_batch})")

    def _is_chunk_complete(self, data_type: str, chunk_id: int) -> bool:
        """
        Check if a chunk is complete and ready to be flushed.
        A chunk is complete if it has all expected batches in the current range.
        """
        if self.current_batch_range is None:
            return False

        start_batch, end_batch = self.current_batch_range
        chunk_start = self.get_chunk_start_batch(chunk_id)
        chunk_end = self.get_chunk_end_batch(chunk_id)

        # Determine the actual batch range for this chunk within our processing range
        actual_start = max(chunk_start, start_batch)
        actual_end = min(chunk_end, end_batch)

        if actual_start >= actual_end:
            return False

        # Check if we have all batches in this range
        expected_batches = set(range(actual_start, actual_end))
        stored_batches = set(self._immediate_chunk_storage[data_type][chunk_id].keys())

        return expected_batches.issubset(stored_batches)

    def _flush_chunk_immediately(self, data_type: str, chunk_id: int):
        """
        Immediately flush a complete chunk to disk and clear it from memory.
        """
        with self._storage_locks[data_type]:
            if chunk_id not in self._immediate_chunk_storage[data_type]:
                return

            chunk_data = self._immediate_chunk_storage[data_type].pop(chunk_id)

            # Submit async write
            future = self.executor.submit(
                self._write_chunk_to_disk, data_type, chunk_id, chunk_data
            )
            self.futures.append(future)

    def store_gradients(self, batch_idx: int, gradients: List[torch.Tensor], is_test: bool = False) -> None:
        """Store gradients for a batch using immediate flushing approach."""
        if is_test:
            return  # Skip test gradients

        # Move to CPU and create dict
        cpu_gradients = [grad.cpu() if grad.device.type != 'cpu' else grad for grad in gradients]
        grad_dict = {idx: grad for idx, grad in enumerate(cpu_gradients)}

        # Store in immediate chunk storage
        data_type = 'gradients'
        chunk_id = self.get_chunk_id(batch_idx)

        with self._storage_locks[data_type]:
            if chunk_id not in self._immediate_chunk_storage[data_type]:
                self._immediate_chunk_storage[data_type][chunk_id] = {}
            self._immediate_chunk_storage[data_type][chunk_id][batch_idx] = grad_dict

        # Check if chunk is complete and flush immediately if so
        if self._is_chunk_complete(data_type, chunk_id):
            self._flush_chunk_immediately(data_type, chunk_id)

    def store_ifvp(self, batch_idx: int, ifvp: List[torch.Tensor]) -> None:
        """Store IFVP for a batch using immediate flushing approach."""
        # Move to CPU and create dict
        cpu_ifvp = [ivp.cpu() if ivp.device.type != 'cpu' else ivp for ivp in ifvp]
        ifvp_dict = {idx: ivp for idx, ivp in enumerate(cpu_ifvp)}

        # Store in immediate chunk storage
        data_type = 'ifvp'
        chunk_id = self.get_chunk_id(batch_idx)

        with self._storage_locks[data_type]:
            if chunk_id not in self._immediate_chunk_storage[data_type]:
                self._immediate_chunk_storage[data_type][chunk_id] = {}
            self._immediate_chunk_storage[data_type][chunk_id][batch_idx] = ifvp_dict

        # Check if chunk is complete and flush immediately if so
        if self._is_chunk_complete(data_type, chunk_id):
            self._flush_chunk_immediately(data_type, chunk_id)

    def finalize_batch_range(self):
        """Write any remaining incomplete chunks for the current batch range."""
        if self.current_batch_range is None:
            return

        start_batch, end_batch = self.current_batch_range
        start_chunk = self.get_chunk_id(start_batch)
        end_chunk = self.get_chunk_id(end_batch - 1) if end_batch > start_batch else start_chunk

        # Flush any remaining chunks (complete or incomplete)
        for data_type in ['gradients', 'ifvp']:
            with self._storage_locks[data_type]:
                chunks_to_flush = []
                for chunk_id in range(start_chunk, end_chunk + 1):
                    if chunk_id in self._immediate_chunk_storage[data_type]:
                        chunks_to_flush.append(chunk_id)

                for chunk_id in chunks_to_flush:
                    chunk_data = self._immediate_chunk_storage[data_type].pop(chunk_id)
                    future = self.executor.submit(
                        self._write_chunk_to_disk, data_type, chunk_id, chunk_data
                    )
                    self.futures.append(future)

        self.current_batch_range = None

    def _write_chunk_to_disk(self, data_type: str, chunk_id: int, chunk_data: Dict):
        """Write a chunk to disk."""
        try:
            subdir = 'grad' if data_type == 'gradients' else data_type
            chunk_dir = os.path.join(self.cache_dir, subdir)

            # Convert to format for memory mapper
            formatted_data = [(batch_idx, batch_dict)
                            for batch_idx, batch_dict in sorted(chunk_data.items())]

            chunk_filename = ChunkedMemoryMapHandler.write_chunk(
                chunk_dir, data_type, formatted_data, dtype='float32'
            )

            logger.debug(f"Wrote {data_type} chunk {chunk_id} with {len(chunk_data)} batches to {chunk_filename}")

        except Exception as e:
            logger.error(f"Error writing {data_type} chunk {chunk_id}: {e}")
            raise

    def store_preconditioner(self, layer_idx: int, preconditioner: torch.Tensor) -> None:
        """Store a preconditioner for a layer on disk."""
        cpu_precond = preconditioner.cpu() if preconditioner.device.type != 'cpu' else preconditioner
        file_path = self.get_path('preconditioners', layer_idx=layer_idx)
        self.save_tensor(cpu_precond, file_path)

    def get_path(self, data_type: DataTypeOptions, batch_idx: Optional[int] = None,
                layer_idx: Optional[int] = None, is_test: bool = False) -> str:
        """Generate standardized path for data storage."""
        if not self.cache_dir:
            raise ValueError("Cache directory is not set")

        if data_type == 'preconditioners':
            if layer_idx is None:
                raise ValueError("Layer index must be provided for preconditioners")
            filename = f"layer_{layer_idx}.pt"
            return os.path.join(self.cache_dir, 'precond', filename)
        else:
            raise ValueError(f"get_path only supports preconditioners, got {data_type}")

    def save_tensor(self, tensor: torch.Tensor, path: str, async_save: bool = True) -> None:
        """Save a tensor to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if tensor.is_cuda:
            cpu_tensor = tensor.cpu()
            if async_save:
                future = self.executor.submit(torch.save, cpu_tensor, path)
                self.futures.append(future)
            else:
                torch.save(cpu_tensor, path)
        else:
            if async_save:
                future = self.executor.submit(torch.save, tensor, path)
                self.futures.append(future)
            else:
                torch.save(tensor, path)

    def load_tensor(self, path: str) -> torch.Tensor:
        """Load a tensor from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find tensor file: {path}")
        return torch.load(path)

    def retrieve_gradients(self, batch_idx: int, is_test: bool = False) -> List[torch.Tensor]:
        """Retrieve gradients for a batch from chunked storage."""
        return self._retrieve_chunked_data('gradients', batch_idx)

    def retrieve_ifvp(self, batch_idx: int) -> List[torch.Tensor]:
        """Retrieve IFVP for a batch from chunked storage."""
        return self._retrieve_chunked_data('ifvp', batch_idx)

    def retrieve_preconditioner(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Retrieve a preconditioner for a layer from disk."""
        file_path = self.get_path('preconditioners', layer_idx=layer_idx)
        if not os.path.exists(file_path):
            return None
        return self.load_tensor(file_path)

    def _retrieve_chunked_data(self, data_type: DataTypeOptions, batch_idx: int) -> List[torch.Tensor]:
        """Retrieve data for a specific batch from chunked storage."""
        # First check if data is in immediate memory buffer
        chunk_id = self.get_chunk_id(batch_idx)

        with self._storage_locks[data_type]:
            if chunk_id in self._immediate_chunk_storage[data_type]:
                if batch_idx in self._immediate_chunk_storage[data_type][chunk_id]:
                    data_dict = self._immediate_chunk_storage[data_type][chunk_id][batch_idx]
                    max_layer = max(data_dict.keys()) if data_dict else -1
                    result = []
                    for layer_idx in range(max_layer + 1):
                        if layer_idx in data_dict:
                            result.append(data_dict[layer_idx])
                        else:
                            result.append(torch.tensor([]))
                    return result

        # Search in chunk files on disk
        subdir = self._get_chunk_subdir(data_type)
        chunk_path = os.path.join(self.cache_dir, subdir)

        chunk_files = ChunkedMemoryMapHandler.find_chunk_files(chunk_path, data_type)

        for chunk_filename in chunk_files:
            try:
                metadata = ChunkedMemoryMapHandler.read_chunk_metadata(chunk_path, chunk_filename)

                # Check if batch exists in this chunk
                batch_found = any(batch_info["batch_idx"] == batch_idx
                                for batch_info in metadata["batches"])

                if batch_found:
                    data_dict = ChunkedMemoryMapHandler.load_chunk_batch_dict(
                        chunk_path, chunk_filename, batch_idx)

                    max_layer = max(data_dict.keys()) if data_dict else -1
                    result = []
                    for layer_idx in range(max_layer + 1):
                        if layer_idx in data_dict:
                            result.append(data_dict[layer_idx])
                        else:
                            result.append(torch.tensor([]))
                    return result

            except Exception as e:
                logger.warning(f"Error reading chunk {chunk_filename}: {e}")
                continue

        logger.warning(f"Batch {batch_idx} not found for data type {data_type}")
        return [torch.tensor([])]

    def _get_chunk_subdir(self, data_type: DataTypeOptions) -> str:
        """Get subdirectory for a data type."""
        if data_type == 'gradients':
            return 'grad'
        elif data_type == 'preconditioners':
            return 'precond'
        elif data_type == 'ifvp':
            return 'ifvp'
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def create_gradient_dataloader(self, data_type: str, batch_size: int = 1,
                                num_workers: int = 0, pin_memory: bool = True,
                                batch_range: Optional[Tuple[int, int]] = None,
                                is_test: bool = False) -> Optional[torch.utils.data.DataLoader]:
        """Create a DataLoader for loading chunked data from disk."""
        try:
            from ..data.dataset import create_chunked_dataloader

            return create_chunked_dataloader(
                disk_io=self,
                data_type=data_type,
                batch_size=batch_size,
                num_workers=0,  # Always 0
                pin_memory=pin_memory,
                batch_range=batch_range,
                is_test=is_test,
                use_chunk_dataset=True
            )
        except ImportError:
            logger.error("Failed to import dataset modules")
            return None

    def find_batch_files(self, data_type: DataTypeOptions, is_test: bool = False) -> List[str]:
        """Find all chunk files for a specific data type."""
        if not self.cache_dir:
            return []

        subdir = self._get_chunk_subdir(data_type)
        chunk_path = os.path.join(self.cache_dir, subdir)

        chunk_files = ChunkedMemoryMapHandler.find_chunk_files(chunk_path, data_type)
        return [os.path.join(chunk_path, f"{chunk_file}.mmap") for chunk_file in chunk_files]

    def wait_for_async_operations(self) -> None:
        """Wait for all pending async operations to complete."""
        for future in self.futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Async operation failed: {e}")
        self.futures = []

    def has_preconditioners(self) -> bool:
        """Check if preconditioners are available on disk."""
        if not self.cache_dir:
            return False
        precond_dir = os.path.join(self.cache_dir, 'precond')
        if not os.path.exists(precond_dir):
            return False
        return len([f for f in os.listdir(precond_dir) if f.endswith('.pt')]) > 0

    def has_ifvp(self) -> bool:
        """Check if IFVP are available on disk."""
        if not self.cache_dir:
            return False
        ifvp_dir = os.path.join(self.cache_dir, 'ifvp')
        if not os.path.exists(ifvp_dir):
            return False
        chunk_files = ChunkedMemoryMapHandler.find_chunk_files(ifvp_dir, 'ifvp')
        return len(chunk_files) > 0

    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'executor'):
            # Flush any remaining data before cleanup
            try:
                for data_type in ['gradients', 'ifvp']:
                    with self._storage_locks[data_type]:
                        chunks_to_flush = list(self._immediate_chunk_storage[data_type].keys())
                        for chunk_id in chunks_to_flush:
                            chunk_data = self._immediate_chunk_storage[data_type].pop(chunk_id)
                            if chunk_data:  # Only flush if there's data
                                future = self.executor.submit(
                                    self._write_chunk_to_disk, data_type, chunk_id, chunk_data
                                )
                                self.futures.append(future)
                                logger.warning(f"Emergency flush of {data_type} chunk {chunk_id} during cleanup")
            except Exception as e:
                logger.error(f"Error during emergency flush: {e}")

            self.wait_for_async_operations()
            self.executor.shutdown(wait=True)