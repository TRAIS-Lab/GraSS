"""
Enhanced dataset classes for loading chunked gradients and influence function data.
"""

# Standard library imports
import os
import re
import gc
import logging
from typing import List, Optional, Tuple, Dict, Any
from collections import defaultdict

# Third-party imports
import torch
import torch.utils.data

# Configure logger
logger = logging.getLogger(__name__)

# Forward declaration for lazy imports to avoid circular imports
ChunkedMemoryMapHandler = None

def _lazy_import_memory_map():
    """Lazy import of memory map module to avoid circular imports"""
    global ChunkedMemoryMapHandler
    if ChunkedMemoryMapHandler is None:
        try:
            from ..io.memory_map import ChunkedMemoryMapHandler
        except ImportError:
            logger.warning("Failed to import ChunkedMemoryMapHandler")
            # If that fails, the user must ensure the correct import path

def extract_batch_range_from_filename(chunk_filename):
    """
    Utility function to extract batch range from chunk filename.

    Args:
        chunk_filename: Chunk filename (e.g., "chunk_gradients_0_31")

    Returns:
        Tuple of (batch_start, batch_end) or None if extraction fails
    """
    try:
        # Parse chunk_gradients_0_31 -> (0, 31)
        parts = chunk_filename.split('_')
        if len(parts) >= 4 and parts[0] == 'chunk':
            batch_start = int(parts[2])
            batch_end = int(parts[3])
            return (batch_start, batch_end)
        return None
    except (ValueError, IndexError):
        logger.warning(f"Could not extract batch range from {chunk_filename}")
        return None

def custom_collate_fn(batch):
    """
    Custom collate function for chunked gradient datasets.
    Instead of trying to stack tensors, it returns lists of batch indices and dictionaries.

    Args:
        batch: List of (batch_indices, batch_dicts) tuples from the dataset

    Returns:
        Tuple of (all_batch_indices, all_batch_dicts) where:
        - all_batch_indices is a flattened list of all batch indices
        - all_batch_dicts is a flattened list of all dictionaries containing gradients or IFVP data
    """
    all_batch_indices = []
    all_batch_dicts = []

    for batch_indices, batch_dicts in batch:
        all_batch_indices.extend(batch_indices)
        all_batch_dicts.extend(batch_dicts)

    return all_batch_indices, all_batch_dicts

class ChunkedGradientDataset(torch.utils.data.Dataset):
    """Enhanced dataset for loading gradient or IFVP chunk files from disk."""

    def __init__(self, disk_io, data_type="gradients", batch_range=None, is_test=False, layer_names=None):
        """
        Initialize dataset for loading chunked gradient or IFVP files.

        Args:
            disk_io: ChunkedDiskIOManager instance
            data_type: Type of data to load ("gradients" or "ifvp")
            batch_range: Optional tuple of (start_batch, end_batch) to filter batches
            is_test: Whether to load test data files
            layer_names: List of layer names for optional validation
        """
        self.disk_io = disk_io
        self.data_type = data_type
        self.is_test = is_test
        self.layer_names = layer_names
        self.batch_range = batch_range

        # Get chunk information
        self.chunk_info = self._load_chunk_info()

        logger.info(f"Found {len(self.chunk_info)} chunks for {data_type}")

    def safe_extract_batch_range(self, chunk_filename):
        """
        Extract batch range from chunk filename.

        Args:
            chunk_filename: Chunk filename (e.g., "chunk_gradients_0_31")

        Returns:
            Tuple of (batch_start, batch_end) or None if extraction fails
        """
        return extract_batch_range_from_filename(chunk_filename)

    def _load_chunk_info(self) -> List[Dict[str, Any]]:
        """
        Load information about all available chunks.

        Returns:
            List of chunk information dictionaries
        """
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

                # Filter batches based on batch_range if specified
                filtered_batches = []
                if self.batch_range is not None:
                    start_batch, end_batch = self.batch_range
                    for batch_info in metadata["batches"]:
                        batch_idx = batch_info["batch_idx"]
                        if start_batch <= batch_idx < end_batch:
                            filtered_batches.append(batch_info)
                else:
                    filtered_batches = metadata["batches"]

                if filtered_batches:  # Only add chunks that have relevant batches
                    chunk_info.append({
                        "chunk_filename": chunk_filename,
                        "chunk_path": chunk_path,
                        "metadata": metadata,
                        "filtered_batches": filtered_batches,
                        "batch_range": self.safe_extract_batch_range(chunk_filename)
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
        Get a chunk of data with proper cleanup to prevent file handle leaks.

        Args:
            idx: Index of the chunk

        Returns:
            Tuple of (batch_indices, batch_dicts) where:
            - batch_indices is a list of batch indices in this chunk
            - batch_dicts is a list of dictionaries containing the data for each batch
        """
        if idx >= len(self.chunk_info):
            raise IndexError(f"Index {idx} out of range for {len(self.chunk_info)} chunks")

        chunk_info = self.chunk_info[idx]
        chunk_filename = chunk_info["chunk_filename"]
        chunk_path = chunk_info["chunk_path"]
        filtered_batches = chunk_info["filtered_batches"]

        # Load all batches from this chunk with proper cleanup
        _lazy_import_memory_map()
        try:
            all_batch_data = ChunkedMemoryMapHandler.load_chunk_all_batches(chunk_path, chunk_filename)

            # Extract only the filtered batches
            batch_indices = []
            batch_dicts = []

            for batch_info in filtered_batches:
                batch_idx = batch_info["batch_idx"]
                if batch_idx in all_batch_data:
                    batch_indices.append(batch_idx)
                    batch_dicts.append(all_batch_data[batch_idx])

            # Force cleanup of intermediate data
            del all_batch_data
            gc.collect()

            return batch_indices, batch_dicts

        except Exception as e:
            logger.error(f"Error loading chunk {chunk_filename}: {e}")
            # Return empty data on error
            return [], []

class ChunkedBatchDataset(torch.utils.data.Dataset):
    """
    Alternative dataset that yields individual batches from chunks.
    Useful when you want to iterate over individual batches rather than chunks.
    """

    def __init__(self, disk_io, data_type="gradients", batch_range=None, is_test=False, layer_names=None):
        """
        Initialize dataset for loading individual batches from chunked files.

        Args:
            disk_io: ChunkedDiskIOManager instance
            data_type: Type of data to load ("gradients" or "ifvp")
            batch_range: Optional tuple of (start_batch, end_batch) to filter batches
            is_test: Whether to load test data files
            layer_names: List of layer names for optional validation
        """
        self.disk_io = disk_io
        self.data_type = data_type
        self.is_test = is_test
        self.layer_names = layer_names
        self.batch_range = batch_range

        # Build index of all available batches
        self.batch_index = self._build_batch_index()

        logger.info(f"Found {len(self.batch_index)} individual batches for {data_type}")

    def _build_batch_index(self) -> List[Dict[str, Any]]:
        """
        Build an index of all available batches across all chunks.

        Returns:
            List of batch information dictionaries
        """
        batch_index = []

        # Get subdirectory
        subdir = self.disk_io._get_chunk_subdir(self.data_type)
        chunk_path = os.path.join(self.disk_io.cache_dir, subdir)

        if not os.path.exists(chunk_path):
            return batch_index

        # Find all chunk files
        _lazy_import_memory_map()
        chunk_files = ChunkedMemoryMapHandler.find_chunk_files(chunk_path, self.data_type)

        for chunk_filename in chunk_files:
            try:
                # Load chunk metadata
                metadata = ChunkedMemoryMapHandler.read_chunk_metadata(chunk_path, chunk_filename)

                # Add each batch to the index
                for batch_info in metadata["batches"]:
                    batch_idx = batch_info["batch_idx"]

                    # Filter by batch_range if specified
                    if self.batch_range is not None:
                        start_batch, end_batch = self.batch_range
                        if not (start_batch <= batch_idx < end_batch):
                            continue

                    batch_index.append({
                        "batch_idx": batch_idx,
                        "chunk_filename": chunk_filename,
                        "chunk_path": chunk_path
                    })

            except Exception as e:
                logger.warning(f"Error loading chunk {chunk_filename}: {e}")
                continue

        # Sort by batch index for consistency
        batch_index.sort(key=lambda x: x["batch_idx"])
        return batch_index

    def __len__(self):
        return len(self.batch_index)

    def __getitem__(self, idx):
        """
        Get a single batch of data with proper cleanup.

        Args:
            idx: Index of the batch

        Returns:
            Tuple of (batch_idx, batch_dict) where:
            - batch_idx is the batch index
            - batch_dict is a dictionary containing the data for this batch
        """
        if idx >= len(self.batch_index):
            raise IndexError(f"Index {idx} out of range for {len(self.batch_index)} batches")

        batch_info = self.batch_index[idx]
        batch_idx = batch_info["batch_idx"]
        chunk_filename = batch_info["chunk_filename"]
        chunk_path = batch_info["chunk_path"]

        # Load the specific batch from the chunk with proper cleanup
        _lazy_import_memory_map()
        try:
            batch_dict = ChunkedMemoryMapHandler.load_chunk_batch_dict(chunk_path, chunk_filename, batch_idx)
            return batch_idx, batch_dict

        except Exception as e:
            logger.error(f"Error loading batch {batch_idx} from chunk {chunk_filename}: {e}")
            # Return empty data on error
            return batch_idx, {}

def create_chunked_dataloader(disk_io, data_type="gradients", batch_size=1,
                             pin_memory=True, batch_range=None, is_test=False,
                             use_chunk_dataset=True) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for chunked data with optimizations to prevent file handle exhaustion.

    Args:
        disk_io: ChunkedDiskIOManager instance
        data_type: Type of data to load ("gradients" or "ifvp")
        batch_size: Batch size for the DataLoader
        pin_memory: Whether to pin memory
        batch_range: Optional range of batches to include
        is_test: Whether to load test data
        use_chunk_dataset: If True, use ChunkedGradientDataset (yields chunks),
                          if False, use ChunkedBatchDataset (yields individual batches)

    Returns:
        DataLoader for efficient loading of chunked data
    """
    if use_chunk_dataset:
        dataset = ChunkedGradientDataset(
            disk_io=disk_io,
            data_type=data_type,
            batch_range=batch_range,
            is_test=is_test
        )
        collate_fn = custom_collate_fn
    else:
        dataset = ChunkedBatchDataset(
            disk_io=disk_io,
            data_type=data_type,
            batch_range=batch_range,
            is_test=is_test
        )
        collate_fn = lambda batch: ([item[0] for item in batch], [item[1] for item in batch])

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=pin_memory,
        shuffle=False,
        collate_fn=collate_fn
    )