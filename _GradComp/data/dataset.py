"""
Optimized dataset classes for loading pure tensor-based chunked data.
"""

import os
import gc
import logging
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.utils.data

logger = logging.getLogger(__name__)

# Forward declaration for lazy imports
ChunkedMemoryMapHandler = None

def _lazy_import_memory_map():
    """Lazy import of memory map module to avoid circular imports"""
    global ChunkedMemoryMapHandler
    if ChunkedMemoryMapHandler is None:
        try:
            from ..io.memory_map import ChunkedMemoryMapHandler
        except ImportError:
            logger.warning("Failed to import ChunkedMemoryMapHandler")

class TensorChunkedDataset(torch.utils.data.Dataset):
    """
    Optimized dataset that loads chunks as pure concatenated tensors.
    """

    def __init__(self, disk_io, data_type="gradients", batch_range=None):
        """
        Initialize dataset for loading tensor-based chunks.

        Args:
            disk_io: ChunkedDiskIOManager instance
            data_type: Type of data to load ("gradients" or "ifvp")
            batch_range: Optional tuple of (start_batch, end_batch) to filter batches
        """
        self.disk_io = disk_io
        self.data_type = data_type
        self.batch_range = batch_range

        # Get chunk information
        self.chunk_info = self._load_chunk_info()

        logger.debug(f"TensorChunkedDataset: Found {len(self.chunk_info)} chunks for {data_type}")

    def _load_chunk_info(self) -> List[Dict[str, Any]]:
        """Load information about all available chunks."""
        chunk_info = []

        # Get subdirectory
        subdir = self.disk_io._get_chunk_subdir(self.data_type)
        chunk_path = os.path.join(self.disk_io.cache_dir, subdir)

        if not os.path.exists(chunk_path):
            return chunk_info

        # Find all chunk files
        _lazy_import_memory_map()
        chunk_files = ChunkedMemoryMapHandler.find_chunk_files(chunk_path, self.data_type)

        for chunk_filename in chunk_files:
            try:
                # Load chunk metadata
                metadata = ChunkedMemoryMapHandler.read_chunk_metadata(chunk_path, chunk_filename)

                # Check if chunk contains batches in our range
                if self.batch_range is not None:
                    start_batch, end_batch = self.batch_range
                    chunk_batches = [b["batch_idx"] for b in metadata["batches"]]

                    # Skip chunks that don't overlap with our range
                    if not any(start_batch <= idx < end_batch for idx in chunk_batches):
                        continue

                chunk_info.append({
                    "chunk_filename": chunk_filename,
                    "chunk_path": chunk_path,
                    "metadata": metadata
                })

            except Exception as e:
                logger.warning(f"Error loading chunk {chunk_filename}: {e}")
                continue

        # Sort by chunk filename for consistency
        chunk_info.sort(key=lambda x: x["chunk_filename"])
        return chunk_info

    def __len__(self):
        return len(self.chunk_info)

    def __getitem__(self, idx):
        """
        Get a chunk as a concatenated tensor.

        Returns:
            Tuple of (tensor, batch_mapping) where:
            - tensor has shape (total_samples_in_chunk, total_proj_dim)
            - batch_mapping maps batch_idx to (start_row, end_row) in tensor
        """
        if idx >= len(self.chunk_info):
            raise IndexError(f"Index {idx} out of range for {len(self.chunk_info)} chunks")

        chunk_info = self.chunk_info[idx]
        chunk_filename = chunk_info["chunk_filename"]
        chunk_path = chunk_info["chunk_path"]

        _lazy_import_memory_map()
        try:
            # Load chunk as tensor with batch mapping
            tensor, batch_mapping = ChunkedMemoryMapHandler.load_chunk_batch_range(
                chunk_path, chunk_filename, self.batch_range
            )

            return tensor, batch_mapping

        except Exception as e:
            logger.error(f"Error loading chunk {chunk_filename}: {e}")
            # Return empty data on error
            layer_dims = chunk_info["metadata"].get("layer_dims", [])
            total_proj_dim = sum(layer_dims) if layer_dims else 0
            empty_tensor = torch.empty(0, total_proj_dim)
            return empty_tensor, {}

def tensor_collate_fn(batch):
    """
    Collate function for tensor chunks.

    Args:
        batch: List of (tensor, batch_mapping) tuples

    Returns:
        Combined tensor and mapping
    """
    if len(batch) == 1:
        return batch[0]

    # Combine multiple chunks
    tensors = []
    combined_mapping = {}
    row_offset = 0

    for tensor, batch_mapping in batch:
        tensors.append(tensor)

        # Update mapping with new offsets
        for batch_idx, (start, end) in batch_mapping.items():
            batch_size = end - start
            combined_mapping[batch_idx] = (row_offset + start, row_offset + start + batch_size)

        row_offset += tensor.shape[0]

    # Concatenate tensors
    combined_tensor = torch.cat(tensors, dim=0)

    return combined_tensor, combined_mapping

def create_tensor_dataloader(disk_io, data_type="gradients", batch_size=1,
                            pin_memory=True, batch_range=None, is_test=False,
                            num_workers=0) -> torch.utils.data.DataLoader:
    """
    Create an optimized DataLoader for tensor-based chunked data.

    Args:
        disk_io: ChunkedDiskIOManager instance
        data_type: Type of data to load ("gradients" or "ifvp")
        batch_size: Number of chunks to load at once
        pin_memory: Whether to pin memory
        batch_range: Optional range of batches to include
        is_test: Whether to load test data (unused)
        num_workers: Number of workers (always 0)

    Returns:
        DataLoader for efficient loading of chunked tensor data
    """
    dataset = TensorChunkedDataset(
        disk_io=disk_io,
        data_type=data_type,
        batch_range=batch_range
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=pin_memory,
        shuffle=False,
        collate_fn=tensor_collate_fn
    )