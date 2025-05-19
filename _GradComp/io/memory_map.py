"""
Memory-mapped file operations for efficient tensor storage and access.
"""

import json
import os
import numpy as np
import torch
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any, Optional
import logging

# Configure logger
logger = logging.getLogger(__name__)

class MemoryMapHandler:
    """
    Handler for memory-mapped file operations to efficiently read and write tensor data.
    Uses memory-mapped binary files for tensor data and JSON files for metadata.
    """

    @staticmethod
    def write(save_path: str, filename: str, batch_idx: int,
              tensors: Dict[int, torch.Tensor], dtype: str = 'float32') -> None:
        """
        Write tensors to a memory-mapped file with metadata.

        Args:
            save_path: Directory to save files
            filename: Base name for the memory-mapped file
            batch_idx: Batch index for metadata
            tensors: Dictionary mapping layer indices to tensors
            dtype: NumPy data type to use for storage
        """
        # Ensure directory exists
        os.makedirs(save_path, exist_ok=True)

        # Define file paths
        mmap_path = os.path.join(save_path, filename)
        metadata_path = os.path.join(save_path, f"{os.path.splitext(filename)[0]}_metadata.json")

        # Calculate total size needed
        total_size = sum(tensor.numel() for tensor in tensors.values())

        # Create memory-mapped file
        try:
            mmap = np.memmap(mmap_path, dtype=np.dtype(dtype), mode="w+", shape=(total_size,))

            # Prepare metadata
            metadata = {
                "batch_idx": batch_idx,
                "layers": [],
                "offsets": [],
                "shapes": [],
                "dtype": dtype,
                "total_size": total_size
            }

            # Write tensor data and collect metadata
            offset = 0
            for layer_idx, tensor in sorted(tensors.items()):
                # Convert to NumPy and write to mmap
                array = tensor.cpu().detach().numpy().astype(dtype)
                size = array.size
                shape = array.shape

                # Write to memory-mapped file
                mmap[offset:offset+size] = array.ravel()

                # Update metadata
                metadata["layers"].append(int(layer_idx))
                metadata["offsets"].append(int(offset))
                metadata["shapes"].append(list(shape))

                # Update offset for next tensor
                offset += size

            # Flush to disk
            mmap.flush()

            # Save metadata
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Successfully wrote memory-mapped file: {mmap_path}")

        except Exception as e:
            logger.error(f"Error writing memory-mapped file {mmap_path}: {str(e)}")
            raise

    @staticmethod
    @contextmanager
    def read(path: str, filename: str, dtype: str = 'float32'):
        """
        Context manager to read from a memory-mapped file.

        Args:
            path: Directory containing the memory-mapped file
            filename: Base name of the memory-mapped file
            dtype: NumPy data type used for storage

        Yields:
            memory-mapped array object
        """
        mmap_path = os.path.join(path, filename)
        mmap = np.memmap(mmap_path, dtype=np.dtype(dtype), mode="r")
        try:
            yield mmap
        finally:
            del mmap  # Ensure the mmap is closed

    @staticmethod
    def read_metadata(path: str, metadata_filename: str) -> Dict[str, Any]:
        """
        Read metadata from a JSON file.

        Args:
            path: Directory containing the metadata file
            metadata_filename: Name of the metadata file

        Returns:
            Metadata dictionary
        """
        metadata_path = os.path.join(path, metadata_filename)
        with open(metadata_path, "r") as f:
            return json.load(f)

    @staticmethod
    def load_tensor_dict(path: str, filename: str) -> Dict[int, torch.Tensor]:
        """
        Load tensor dictionary from memory-mapped file and metadata.

        Args:
            path: Directory containing the files
            filename: Base name of the memory-mapped file

        Returns:
            Dictionary mapping layer indices to tensors
        """
        # Define file paths
        mmap_path = os.path.join(path, filename)
        metadata_path = os.path.join(path, f"{os.path.splitext(filename)[0]}_metadata.json")

        # Check if files exist
        if not os.path.exists(mmap_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Missing files: {mmap_path} or {metadata_path}")

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Get data type
        dtype = metadata["dtype"]

        # Open memory-mapped file
        with MemoryMapHandler.read(path, filename, dtype) as mmap:
            # Extract all tensors
            result = {}

            for i, layer_idx in enumerate(metadata["layers"]):
                offset = metadata["offsets"][i]
                shape = metadata["shapes"][i]

                # Calculate size
                size = np.prod(shape)

                # Extract array slice and copy to new tensor
                array = np.array(mmap[offset:offset+size]).reshape(shape)
                tensor = torch.from_numpy(array.copy())

                # Store in result dictionary
                result[int(layer_idx)] = tensor

            return result