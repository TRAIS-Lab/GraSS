"""
Enhanced dataset classes for loading tensor-based chunked gradients and influence function data.
"""

import os
import gc
import logging
from typing import List, Optional, Tuple, Dict, Any
from collections import defaultdict

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

def extract_batch_range_from_filename(chunk_filename):
    """Extract batch range from chunk filename."""
    try:
        parts = chunk_filename.split('_')
        if len(parts) >= 4 and parts[0] == 'chunk':
            batch_start = int(parts[2])
            batch_end = int(parts[3])
            return (batch_start, batch_end)
        return None
    except (ValueError, IndexError):
        logger.warning(f"Could not extract batch range from {chunk_filename}")
        return None

def tensor_collate_fn(batch):
    """
    Custom collate function for tensor-based chunked datasets.
    Returns concatenated tensors and batch mappings.

    Args:
        batch: List of (tensor, batch_mapping) tuples from the dataset

    Returns:
        Tuple of (concatenated_tensor, combined_batch_mapping)
    """
    all_tensors = []
    combined_mapping = {}
    offset = 0

    for tensor, batch_mapping in batch:
        if tensor.numel() > 0:
            all_tensors.append(tensor)

            # Update mapping with offset
            for batch_idx, (start, end) in batch_mapping.items():
                combined_mapping[batch_idx] = (offset + start, offset + end)

            offset += tensor.shape[0]

    if all_tensors:
        concatenated = torch.cat(all_tensors, dim=0)
    else:
        # Return empty tensor with correct dimensions
        concatenated = torch.empty(0, batch[0][0].shape[1] if batch and batch[0][0].dim() > 1 else 0)

    return concatenated, combined_mapping

def dict_collate_fn(batch):
    """
    Collate function that returns batch indices and dictionaries.
    Used for compatibility with existing code.
    """
    all_batch_indices = []
    all_batch_dicts = []

    for batch_indices, batch_dicts in batch:
        all_batch_indices.extend(batch_indices)
        all_batch_dicts.extend(batch_dicts)

    return all_batch_indices, all_batch_dicts

class TensorChunkedDataset(torch.utils.data.Dataset):
    """
    Dataset that loads chunks as concatenated tensors for maximum efficiency.
    """

    def __init__(self, disk_io, data_type="gradients", batch_range=None, layer_names=None):
        """
        Initialize dataset for loading tensor-based chunks.

        Args:
            disk_io: ChunkedDiskIOManager instance
            data_type: Type of data to load ("gradients" or "ifvp")
            batch_range: Optional tuple of (start_batch, end_batch) to filter batches
            layer_names: List of layer names (optional)
        """
        self.disk_io = disk_io
        self.data_type = data_type
        self.batch_range = batch_range
        self.layer_names = layer_names

        # Get chunk information
        self.chunk_info = self._load_chunk_info()

        logger.info(f"TensorChunkedDataset: Found {len(self.chunk_info)} chunks for {data_type}")

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
                    "metadata": metadata,
                    "batch_range": extract_batch_range_from_filename(chunk_filename)
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
            tensor, batch_mapping = ChunkedMemoryMapHandler.load_chunk_as_tensor(
                chunk_path, chunk_filename, self.batch_range
            )

            return tensor, batch_mapping

        except Exception as e:
            logger.error(f"Error loading chunk {chunk_filename}: {e}")
            # Return empty data on error
            empty_tensor = torch.empty(0, chunk_info["metadata"].get("total_proj_dim", 0))
            return empty_tensor, {}

class ChunkedGradientDataset(torch.utils.data.Dataset):
    """
    Legacy dataset for compatibility - loads chunks and returns dictionaries.
    """

    def __init__(self, disk_io, data_type="gradients", batch_range=None, is_test=False, layer_names=None):
        """Initialize dataset for loading chunked gradient or IFVP files."""
        self.disk_io = disk_io
        self.data_type = data_type
        self.is_test = is_test
        self.layer_names = layer_names
        self.batch_range = batch_range

        # Get chunk information
        self.chunk_info = self._load_chunk_info()

        logger.info(f"Found {len(self.chunk_info)} chunks for {data_type}")

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
                metadata = ChunkedMemoryMapHandler.read_chunk_metadata(chunk_path, chunk_filename)

                # Filter batches based on batch_range
                filtered_batches = []
                if self.batch_range is not None:
                    start_batch, end_batch = self.batch_range
                    for batch_info in metadata["batches"]:
                        batch_idx = batch_info["batch_idx"]
                        if start_batch <= batch_idx < end_batch:
                            filtered_batches.append(batch_info)
                else:
                    filtered_batches = metadata["batches"]

                if filtered_batches:
                    chunk_info.append({
                        "chunk_filename": chunk_filename,
                        "chunk_path": chunk_path,
                        "metadata": metadata,
                        "filtered_batches": filtered_batches,
                        "batch_range": extract_batch_range_from_filename(chunk_filename)
                    })

            except Exception as e:
                logger.warning(f"Error loading chunk {chunk_filename}: {e}")
                continue

        chunk_info.sort(key=lambda x: x["chunk_filename"])
        return chunk_info

    def __len__(self):
        return len(self.chunk_info)

    def __getitem__(self, idx):
        """Get a chunk of data as dictionaries for compatibility."""
        if idx >= len(self.chunk_info):
            raise IndexError(f"Index {idx} out of range")

        chunk_info = self.chunk_info[idx]
        chunk_filename = chunk_info["chunk_filename"]
        chunk_path = chunk_info["chunk_path"]
        filtered_batches = chunk_info["filtered_batches"]

        _lazy_import_memory_map()
        try:
            all_batch_data = ChunkedMemoryMapHandler.load_chunk_all_batches(
                chunk_path, chunk_filename
            )

            # Extract only filtered batches
            batch_indices = []
            batch_dicts = []

            for batch_info in filtered_batches:
                batch_idx = batch_info["batch_idx"]
                if batch_idx in all_batch_data:
                    batch_indices.append(batch_idx)
                    batch_dicts.append(all_batch_data[batch_idx])

            del all_batch_data
            gc.collect()

            return batch_indices, batch_dicts

        except Exception as e:
            logger.error(f"Error loading chunk {chunk_filename}: {e}")
            return [], []

def create_chunked_dataloader(disk_io, data_type="gradients", batch_size=1,
                             pin_memory=True, batch_range=None, is_test=False,
                             use_chunk_dataset=True, use_tensor_dataset=False,
                             num_workers=0) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for chunked data.

    Args:
        disk_io: ChunkedDiskIOManager instance
        data_type: Type of data to load ("gradients" or "ifvp")
        batch_size: Batch size for the DataLoader
        pin_memory: Whether to pin memory
        batch_range: Optional range of batches to include
        is_test: Whether to load test data
        use_chunk_dataset: If True, use ChunkedGradientDataset (legacy)
        use_tensor_dataset: If True, use TensorChunkedDataset (efficient)
        num_workers: Number of workers (always 0)

    Returns:
        DataLoader for efficient loading of chunked data
    """
    if use_tensor_dataset:
        dataset = TensorChunkedDataset(
            disk_io=disk_io,
            data_type=data_type,
            batch_range=batch_range
        )
        collate_fn = tensor_collate_fn
    else:
        dataset = ChunkedGradientDataset(
            disk_io=disk_io,
            data_type=data_type,
            batch_range=batch_range,
            is_test=is_test
        )
        collate_fn = dict_collate_fn

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # Always 0
        pin_memory=pin_memory,
        shuffle=False,
        collate_fn=collate_fn
    )