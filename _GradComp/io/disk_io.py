"""
Disk I/O manager.
"""

import os
import threading
from typing import List, Dict, Optional, Literal, Any, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from .memory_map import ChunkedMemoryMapHandler

import logging
logger = logging.getLogger(__name__)

DataTypeOptions = Literal["gradients", "preconditioners", "ifvp"]
HessianOptions = Literal["none", "raw", "kfac", "ekfac"]

@dataclass
class ChunkBuffer:
    """Buffer for accumulating data in tensor format."""
    tensor: torch.Tensor  # Pre-allocated tensor
    batch_indices: List[int]  # Batch indices in this chunk
    batch_info: List[Dict[str, Any]]  # Batch metadata (batch_idx, start_row, end_row)
    current_row: int  # Next row to write to

class ChunkedDiskIOManager:
    """
    Disk I/O manager with pure tensor storage per chunk.
    """

    def __init__(self, cache_dir: str, setting: str, num_threads: int = 16,
                 hessian: HessianOptions = "raw", chunk_size: int = 32,
                 max_samples_per_chunk: int = 2048):
        self.cache_dir = cache_dir
        self.setting = setting
        self.num_threads = num_threads
        self.hessian = hessian
        self.chunk_size = chunk_size
        self.max_samples_per_chunk = max_samples_per_chunk

        # Create cache directory structure
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            for subdir in ['grad', 'ifvp', 'precond']:
                os.makedirs(os.path.join(cache_dir, subdir), exist_ok=True)

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.futures = []

        # Direct tensor accumulation buffers
        self._chunk_buffers = {}  # (data_type, chunk_id) -> ChunkBuffer
        self._buffer_locks = defaultdict(threading.Lock)

        # Track dimensions and batch range
        self.layer_dims = None
        self.total_proj_dim = None
        self.current_batch_range = None

        # Try to load layer dimensions from existing data
        self._load_layer_dims_from_metadata()

        logger.debug(f"Initialized ChunkedDiskIOManager with chunk_size={chunk_size}")

    def get_chunk_id(self, batch_idx: int) -> int:
        """Get chunk ID for a batch index."""
        return batch_idx // self.chunk_size

    def _load_layer_dims_from_metadata(self):
        """Try to load layer dimensions from existing chunk metadata."""
        if not self.cache_dir:
            return

        # Try gradient chunks first
        for data_type, subdir in [('gradients', 'grad'), ('ifvp', 'ifvp')]:
            data_dir = os.path.join(self.cache_dir, subdir)
            if os.path.exists(data_dir):
                try:
                    chunk_files = ChunkedMemoryMapHandler.find_chunk_files(data_dir, data_type)
                    if chunk_files:
                        metadata = ChunkedMemoryMapHandler.read_chunk_metadata(data_dir, chunk_files[0])
                        if 'layer_dims' in metadata:
                            self.layer_dims = metadata['layer_dims']
                            self.total_proj_dim = sum(self.layer_dims)
                            logger.info(f"Loaded layer dimensions from {data_type} metadata: {self.layer_dims}")
                            return
                except Exception as e:
                    logger.debug(f"Could not load layer dims from {data_type}: {e}")

    def _ensure_layer_dims(self):
        """Ensure layer dimensions are available."""
        if self.layer_dims is None:
            self._load_layer_dims_from_metadata()

        if self.layer_dims is None:
            raise ValueError(
                "Layer dimensions not available. Either compute gradients first or "
                "ensure existing gradient/IFVP chunks are present in the cache directory."
            )

    def start_batch_range(self, start_batch: int, end_batch: int):
        """Start processing a batch range."""
        if start_batch % self.chunk_size != 0:
            raise ValueError(f"Batch range start {start_batch} must be aligned to chunk_size {self.chunk_size}")

        self.current_batch_range = (start_batch, end_batch)
        logger.info(f"Starting batch range [{start_batch}, {end_batch})")

    def _initialize_chunk_buffer(self, data_type: str, chunk_id: int) -> ChunkBuffer:
        """Initialize a pre-allocated buffer for a chunk."""
        self._ensure_layer_dims()

        # Pre-allocate tensor for the chunk
        buffer_tensor = torch.zeros(
            self.max_samples_per_chunk,
            self.total_proj_dim,
            dtype=torch.float32,
            pin_memory=True  # Pin memory for faster GPU transfers
        )

        return ChunkBuffer(
            tensor=buffer_tensor,
            batch_indices=[],
            batch_info=[],
            current_row=0
        )

    def store_gradients(self, batch_idx: int, gradients: List[torch.Tensor], is_test: bool = False) -> None:
        """Store gradients directly in tensor format."""
        if is_test:
            return  # Skip test gradients for now

        # Detect layer dimensions on first store
        if self.layer_dims is None:
            self.layer_dims = [g.shape[1] if g.numel() > 0 else 0 for g in gradients]
            self.total_proj_dim = sum(self.layer_dims)
            logger.debug(f"Detected layer dimensions: {len(self.layer_dims)} layers, total={self.total_proj_dim}")

        chunk_id = self.get_chunk_id(batch_idx)
        buffer_key = ('gradients', chunk_id)

        with self._buffer_locks[buffer_key]:
            # Get or create buffer
            if buffer_key not in self._chunk_buffers:
                self._chunk_buffers[buffer_key] = self._initialize_chunk_buffer('gradients', chunk_id)

            buffer = self._chunk_buffers[buffer_key]

            # Concatenate gradients for this batch
            batch_size = next((g.shape[0] for g in gradients if g.numel() > 0), 0)
            if batch_size == 0:
                return

            # Build concatenated tensor for this batch
            batch_features = []
            for layer_idx, (grad, dim) in enumerate(zip(gradients, self.layer_dims)):
                if grad.numel() > 0:
                    batch_features.append(grad.cpu())
                else:
                    # Zero padding for missing layers
                    batch_features.append(torch.zeros(batch_size, dim))

            batch_tensor = torch.cat(batch_features, dim=1)

            # Write to buffer
            start_row = buffer.current_row
            end_row = start_row + batch_size

            if end_row > self.max_samples_per_chunk:
                logger.warning(f"Chunk buffer overflow. Consider increasing max_samples_per_chunk")
                # Force flush and start new buffer
                self._flush_chunk_buffer(buffer_key)

                # Recreate buffer and retry
                self._chunk_buffers[buffer_key] = self._initialize_chunk_buffer('gradients', chunk_id)
                buffer = self._chunk_buffers[buffer_key]
                start_row = 0
                end_row = batch_size

            # Direct copy to pre-allocated buffer
            buffer.tensor[start_row:end_row] = batch_tensor
            buffer.batch_indices.append(batch_idx)
            buffer.batch_info.append({
                "batch_idx": batch_idx,
                "start_row": start_row,
                "end_row": end_row
            })
            buffer.current_row = end_row

            # Check if chunk is complete
            if self._is_chunk_complete('gradients', chunk_id):
                self._flush_chunk_buffer(buffer_key)

    def store_ifvp(self, batch_idx: int, ifvp: List[torch.Tensor]) -> None:
        """Store IFVP directly in tensor format."""
        self._ensure_layer_dims()

        chunk_id = self.get_chunk_id(batch_idx)
        buffer_key = ('ifvp', chunk_id)

        with self._buffer_locks[buffer_key]:
            # Get or create buffer
            if buffer_key not in self._chunk_buffers:
                self._chunk_buffers[buffer_key] = self._initialize_chunk_buffer('ifvp', chunk_id)

            buffer = self._chunk_buffers[buffer_key]

            # Concatenate IFVP for this batch
            batch_size = next((v.shape[0] for v in ifvp if v.numel() > 0), 0)
            if batch_size == 0:
                return

            # Build concatenated tensor
            batch_features = []
            for layer_idx, (vec, dim) in enumerate(zip(ifvp, self.layer_dims)):
                if vec.numel() > 0:
                    batch_features.append(vec.cpu())
                else:
                    batch_features.append(torch.zeros(batch_size, dim))

            batch_tensor = torch.cat(batch_features, dim=1)

            # Write to buffer
            start_row = buffer.current_row
            end_row = start_row + batch_size

            if end_row > self.max_samples_per_chunk:
                self._flush_chunk_buffer(buffer_key)
                self._chunk_buffers[buffer_key] = self._initialize_chunk_buffer('ifvp', chunk_id)
                buffer = self._chunk_buffers[buffer_key]
                start_row = 0
                end_row = batch_size

            buffer.tensor[start_row:end_row] = batch_tensor
            buffer.batch_indices.append(batch_idx)
            buffer.batch_info.append({
                "batch_idx": batch_idx,
                "start_row": start_row,
                "end_row": end_row
            })
            buffer.current_row = end_row

            if self._is_chunk_complete('ifvp', chunk_id):
                self._flush_chunk_buffer(buffer_key)

    def _is_chunk_complete(self, data_type: str, chunk_id: int) -> bool:
        """Check if all batches in a chunk have been stored."""
        if self.current_batch_range is None:
            return False

        start_batch, end_batch = self.current_batch_range
        chunk_start = chunk_id * self.chunk_size
        chunk_end = (chunk_id + 1) * self.chunk_size

        # Expected batches in this chunk
        expected_start = max(chunk_start, start_batch)
        expected_end = min(chunk_end, end_batch)

        if expected_start >= expected_end:
            return False

        # Check buffer
        buffer_key = (data_type, chunk_id)
        if buffer_key not in self._chunk_buffers:
            return False

        buffer = self._chunk_buffers[buffer_key]
        stored_batches = set(buffer.batch_indices)
        expected_batches = set(range(expected_start, expected_end))

        return expected_batches.issubset(stored_batches)

    def _flush_chunk_buffer(self, buffer_key: Tuple[str, int]):
        """Write chunk buffer to disk and clear it."""
        if buffer_key not in self._chunk_buffers:
            return

        buffer = self._chunk_buffers.pop(buffer_key)
        data_type, chunk_id = buffer_key

        # Only write the filled portion
        filled_tensor = buffer.tensor[:buffer.current_row]

        # Submit async write
        future = self.executor.submit(
            self._write_chunk_tensor,
            data_type,
            chunk_id,
            filled_tensor,
            buffer.batch_info
        )
        self.futures.append(future)

    def _write_chunk_tensor(self, data_type: str, chunk_id: int, tensor: torch.Tensor,
                           batch_info: List[Dict[str, Any]]):
        """Write a chunk tensor to disk."""
        try:
            subdir = 'grad' if data_type == 'gradients' else data_type
            chunk_dir = os.path.join(self.cache_dir, subdir)

            # Write using memory map handler
            ChunkedMemoryMapHandler.write_chunk(
                chunk_dir,
                data_type,
                tensor,
                batch_info,
                self.layer_dims,
                dtype='float32'
            )

            logger.debug(f"Wrote {data_type} chunk {chunk_id}: shape={tensor.shape}")

        except Exception as e:
            logger.error(f"Error writing {data_type} chunk {chunk_id}: {e}")
            raise

    def finalize_batch_range(self):
        """Flush any remaining buffers."""
        if self.current_batch_range is None:
            return

        # Flush all remaining buffers
        remaining_buffers = list(self._chunk_buffers.keys())
        for buffer_key in remaining_buffers:
            with self._buffer_locks[buffer_key]:
                self._flush_chunk_buffer(buffer_key)

        self.current_batch_range = None

    def retrieve_gradients(self, batch_idx: int, is_test: bool = False) -> List[torch.Tensor]:
        """Retrieve gradients for a batch and split into layers."""
        return self._retrieve_batch_data('gradients', batch_idx)

    def retrieve_ifvp(self, batch_idx: int) -> List[torch.Tensor]:
        """Retrieve IFVP for a batch and split into layers."""
        return self._retrieve_batch_data('ifvp', batch_idx)

    def _retrieve_batch_data(self, data_type: str, batch_idx: int) -> List[torch.Tensor]:
        """Retrieve data for a specific batch from tensor storage."""
        self._ensure_layer_dims()

        # Check if in current buffer
        chunk_id = self.get_chunk_id(batch_idx)
        buffer_key = (data_type, chunk_id)

        with self._buffer_locks[buffer_key]:
            if buffer_key in self._chunk_buffers:
                buffer = self._chunk_buffers[buffer_key]
                # Find batch in buffer
                for info in buffer.batch_info:
                    if info["batch_idx"] == batch_idx:
                        start_row = info["start_row"]
                        end_row = info["end_row"]
                        batch_tensor = buffer.tensor[start_row:end_row]
                        return self._split_tensor_to_layers(batch_tensor)

        # Load from disk
        subdir = self._get_chunk_subdir(data_type)
        chunk_path = os.path.join(self.cache_dir, subdir)

        chunk_files = ChunkedMemoryMapHandler.find_chunk_files(chunk_path, data_type)

        for chunk_filename in chunk_files:
            batch_slice = ChunkedMemoryMapHandler.load_batch_slice(
                chunk_path, chunk_filename, batch_idx
            )
            if batch_slice is not None:
                return self._split_tensor_to_layers(batch_slice)

        # Return empty tensors
        return [torch.tensor([]) for _ in range(len(self.layer_dims))]

    def _split_tensor_to_layers(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Split concatenated tensor back into per-layer tensors."""
        result = []
        start_idx = 0
        for dim in self.layer_dims:
            end_idx = start_idx + dim
            result.append(tensor[:, start_idx:end_idx].contiguous())
            start_idx = end_idx

        return result

    def create_gradient_dataloader(self, data_type: str, batch_size: int = 1,
                                pin_memory: bool = True, batch_range: Optional[Tuple[int, int]] = None,
                                is_test: bool = False) -> Optional[torch.utils.data.DataLoader]:
        """Create a DataLoader for loading chunked data."""
        try:
            from ..data.dataset import create_tensor_dataloader
            return create_tensor_dataloader(
                disk_io=self,
                data_type=data_type,
                batch_size=batch_size,
                pin_memory=pin_memory,
                batch_range=batch_range,
                is_test=is_test
            )
        except ImportError:
            logger.error("Failed to import dataset modules")
            return None

    def store_preconditioner(self, layer_idx: int, preconditioner: Optional[torch.Tensor]) -> None:
        """Store a preconditioner for a layer."""
        if preconditioner is None:
            return

        cpu_precond = preconditioner.cpu() if preconditioner.device.type != 'cpu' else preconditioner
        file_path = os.path.join(self.cache_dir, 'precond', f'layer_{layer_idx}.pt')

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        future = self.executor.submit(torch.save, cpu_precond, file_path)
        self.futures.append(future)

    def retrieve_preconditioner(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Retrieve a preconditioner for a layer."""
        file_path = os.path.join(self.cache_dir, 'precond', f'layer_{layer_idx}.pt')
        if os.path.exists(file_path):
            return torch.load(file_path)
        return None

    def _get_chunk_subdir(self, data_type: str) -> str:
        """Get subdirectory for a data type."""
        return 'grad' if data_type == 'gradients' else data_type

    def wait_for_async_operations(self):
        """Wait for all async operations to complete."""
        for future in self.futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Async operation failed: {e}")
        self.futures = []

    def has_preconditioners(self) -> bool:
        """Check if preconditioners exist."""
        precond_dir = os.path.join(self.cache_dir, 'precond')
        if not os.path.exists(precond_dir):
            return False
        return any(f.endswith('.pt') for f in os.listdir(precond_dir))

    def has_ifvp(self) -> bool:
        """Check if IFVP data exists."""
        ifvp_dir = os.path.join(self.cache_dir, 'ifvp')
        if not os.path.exists(ifvp_dir):
            return False
        return len(ChunkedMemoryMapHandler.find_chunk_files(ifvp_dir, 'ifvp')) > 0

    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'executor'):
            # Flush remaining buffers
            try:
                remaining = list(self._chunk_buffers.keys())
                for buffer_key in remaining:
                    self._flush_chunk_buffer(buffer_key)
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

            self.wait_for_async_operations()
            self.executor.shutdown(wait=True)