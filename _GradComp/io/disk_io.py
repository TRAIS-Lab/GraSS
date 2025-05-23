"""
Efficient Disk I/O manager with tensor storage per chunk.
"""

import os
import gc
import logging
import time
import threading
from typing import List, Dict, Optional, Literal, Any, Union, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
import numpy as np

from .memory_map import ChunkedMemoryMapHandler

logger = logging.getLogger(__name__)

DataTypeOptions = Literal["gradients", "preconditioners", "ifvp"]
HessianOptions = Literal["none", "raw", "kfac", "ekfac"]

@dataclass
class ChunkBuffer:
    """Buffer for accumulating data in tensor format."""
    tensor: torch.Tensor  # Pre-allocated tensor
    batch_indices: List[int]  # Batch indices in this chunk
    row_positions: List[int]  # Starting row for each batch
    current_row: int  # Next row to write to
    samples_per_batch: List[int]  # Number of samples in each batch

class ChunkedDiskIOManager:
    """
    Disk I/O manager with tensor storage per chunk.
    Maintains chunk-based processing while using efficient tensor format within each chunk.
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
        self.current_batch_range = None

        # Try to load layer dimensions from existing data
        self._load_layer_dims_from_metadata()

        logger.info(f"Initialized ChunkedDiskIOManager with tensor storage, chunk_size={chunk_size}")

    def get_chunk_id(self, batch_idx: int) -> int:
        """Get chunk ID for a batch index."""
        return batch_idx // self.chunk_size

    def _load_layer_dims_from_metadata(self):
        """Try to load layer dimensions from existing chunk metadata."""
        if not self.cache_dir:
            return

        # Try to find layer dims from existing gradient chunks
        grad_dir = os.path.join(self.cache_dir, 'grad')
        if os.path.exists(grad_dir):
            try:
                chunk_files = ChunkedMemoryMapHandler.find_chunk_files(grad_dir, 'gradients')
                if chunk_files:
                    # Load metadata from first chunk
                    metadata = ChunkedMemoryMapHandler.read_chunk_metadata(grad_dir, chunk_files[0])
                    if 'layer_dims' in metadata:
                        self.layer_dims = metadata['layer_dims']
                        logger.info(f"Loaded layer dimensions from gradient metadata: {self.layer_dims}")
                        return
            except Exception as e:
                logger.debug(f"Could not load layer dims from gradients: {e}")

        # Try IFVP chunks if gradients not available
        ifvp_dir = os.path.join(self.cache_dir, 'ifvp')
        if os.path.exists(ifvp_dir):
            try:
                chunk_files = ChunkedMemoryMapHandler.find_chunk_files(ifvp_dir, 'ifvp')
                if chunk_files:
                    metadata = ChunkedMemoryMapHandler.read_chunk_metadata(ifvp_dir, chunk_files[0])
                    if 'layer_dims' in metadata:
                        self.layer_dims = metadata['layer_dims']
                        logger.info(f"Loaded layer dimensions from IFVP metadata: {self.layer_dims}")
                        return
            except Exception as e:
                logger.debug(f"Could not load layer dims from IFVP: {e}")

    def _ensure_layer_dims(self):
        """Ensure layer dimensions are available, loading from metadata if needed."""
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
        # Ensure layer dimensions are available
        self._ensure_layer_dims()

        total_proj_dim = sum(self.layer_dims)

        # Pre-allocate tensor for the chunk
        buffer_tensor = torch.zeros(
            self.max_samples_per_chunk,
            total_proj_dim,
            dtype=torch.float32,
            pin_memory=True  # Pin memory for faster GPU transfers
        )

        return ChunkBuffer(
            tensor=buffer_tensor,
            batch_indices=[],
            row_positions=[],
            current_row=0,
            samples_per_batch=[]
        )

    def store_gradients(self, batch_idx: int, gradients: List[torch.Tensor], is_test: bool = False) -> None:
        """Store gradients directly in tensor format."""
        if is_test:
            return  # Skip test gradients for now

        # Detect layer dimensions on first store
        if self.layer_dims is None:
            self.layer_dims = [g.shape[1] if g.numel() > 0 else 0 for g in gradients]
            logger.info(f"Detected layer dimensions: {self.layer_dims}")

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
            buffer.row_positions.append(start_row)
            buffer.current_row = end_row
            buffer.samples_per_batch.append(batch_size)

            # Check if chunk is complete
            if self._is_chunk_complete('gradients', chunk_id):
                self._flush_chunk_buffer(buffer_key)

    def store_ifvp(self, batch_idx: int, ifvp: List[torch.Tensor]) -> None:
        """Store IFVP directly in tensor format."""
        # Ensure layer dimensions are available (will load from metadata if needed)
        if self.layer_dims is None:
            self._load_layer_dims_from_metadata()

            # If still None, try to infer from IFVP tensors
            if self.layer_dims is None and ifvp:
                self.layer_dims = []
                for vec in ifvp:
                    if vec.numel() > 0:
                        self.layer_dims.append(vec.shape[1] if vec.dim() > 1 else vec.numel())
                    else:
                        self.layer_dims.append(0)
                logger.info(f"Inferred layer dimensions from IFVP: {self.layer_dims}")

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
            buffer.row_positions.append(start_row)
            buffer.current_row = end_row
            buffer.samples_per_batch.append(batch_size)

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
            buffer.batch_indices,
            buffer.row_positions,
            buffer.samples_per_batch
        )
        self.futures.append(future)

    def _write_chunk_tensor(self, data_type: str, chunk_id: int, tensor: torch.Tensor,
                           batch_indices: List[int], row_positions: List[int],
                           samples_per_batch: List[int]):
        """Write a chunk tensor to disk."""
        try:
            subdir = 'grad' if data_type == 'gradients' else data_type
            chunk_dir = os.path.join(self.cache_dir, subdir)

            # Prepare metadata with row mapping
            batch_metadata = []
            for idx, batch_idx in enumerate(batch_indices):
                batch_metadata.append({
                    "batch_idx": batch_idx,
                    "batch_size": samples_per_batch[idx],
                    "start_row": row_positions[idx],
                    "end_row": row_positions[idx] + samples_per_batch[idx]
                })

            # Generate filename
            batch_start = min(batch_indices)
            batch_end = max(batch_indices)
            chunk_filename = f"chunk_{data_type}_{batch_start}_{batch_end}"

            # Write using memory map handler (it handles the format)
            ChunkedMemoryMapHandler.write_chunk(
                chunk_dir,
                data_type,
                [(batch_idx, self._tensor_slice_to_dict(tensor, batch_metadata[i]))
                 for i, batch_idx in enumerate(batch_indices)],
                self.layer_dims,
                dtype='float32'
            )

            logger.debug(f"Wrote {data_type} chunk {chunk_id}: shape={tensor.shape}")

        except Exception as e:
            logger.error(f"Error writing {data_type} chunk {chunk_id}: {e}")
            raise

    def _tensor_slice_to_dict(self, tensor: torch.Tensor, batch_info: Dict) -> Dict[int, torch.Tensor]:
        """Convert a slice of the tensor back to layer dict for compatibility."""
        start_row = batch_info["start_row"]
        end_row = batch_info["end_row"]
        batch_slice = tensor[start_row:end_row]

        result = {}
        start_idx = 0
        for layer_idx, dim in enumerate(self.layer_dims):
            end_idx = start_idx + dim
            result[layer_idx] = batch_slice[:, start_idx:end_idx]
            start_idx = end_idx

        return result

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
        """Retrieve gradients for a batch."""
        return self._retrieve_batch_data('gradients', batch_idx)

    def retrieve_ifvp(self, batch_idx: int) -> List[torch.Tensor]:
        """Retrieve IFVP for a batch."""
        return self._retrieve_batch_data('ifvp', batch_idx)

    def _retrieve_batch_data(self, data_type: str, batch_idx: int) -> List[torch.Tensor]:
        """Retrieve data for a specific batch from tensor storage."""
        # Try to ensure layer dims are loaded
        if self.layer_dims is None:
            self._load_layer_dims_from_metadata()

        # Check if in current buffer
        chunk_id = self.get_chunk_id(batch_idx)
        buffer_key = (data_type, chunk_id)

        with self._buffer_locks[buffer_key]:
            if buffer_key in self._chunk_buffers:
                buffer = self._chunk_buffers[buffer_key]
                if batch_idx in buffer.batch_indices:
                    idx = buffer.batch_indices.index(batch_idx)
                    start_row = buffer.row_positions[idx]
                    batch_size = buffer.samples_per_batch[idx]

                    # Extract and split into layers
                    batch_tensor = buffer.tensor[start_row:start_row + batch_size]
                    return self._split_tensor_to_layers(batch_tensor)

        # Load from disk
        subdir = self._get_chunk_subdir(data_type)
        chunk_path = os.path.join(self.cache_dir, subdir)

        chunk_files = ChunkedMemoryMapHandler.find_chunk_files(chunk_path, data_type)

        for chunk_filename in chunk_files:
            try:
                data_dict = ChunkedMemoryMapHandler.load_chunk_batch_dict(
                    chunk_path, chunk_filename, batch_idx
                )

                # Convert dict to list
                if self.layer_dims:
                    result = []
                    for layer_idx in range(len(self.layer_dims)):
                        if layer_idx in data_dict:
                            result.append(data_dict[layer_idx])
                        else:
                            result.append(torch.tensor([]))
                    return result
                else:
                    # If layer_dims not available, return as is
                    max_idx = max(data_dict.keys()) if data_dict else -1
                    result = []
                    for i in range(max_idx + 1):
                        result.append(data_dict.get(i, torch.tensor([])))
                    return result

            except:
                continue

        # Return empty tensors
        if self.layer_dims:
            return [torch.tensor([]) for _ in range(len(self.layer_dims))]
        else:
            logger.warning(f"Batch {batch_idx} not found and layer dimensions unknown")
            return [torch.tensor([])]

    def _split_tensor_to_layers(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Split concatenated tensor back into per-layer tensors."""
        # Ensure layer dimensions are available
        self._ensure_layer_dims()

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
            from ..data.dataset import create_chunked_dataloader
            return create_chunked_dataloader(
                disk_io=self,
                data_type=data_type,
                batch_size=batch_size,
                pin_memory=pin_memory,
                batch_range=batch_range,
                is_test=is_test,
                use_chunk_dataset=True  # Use chunk-based dataset
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