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

    # @staticmethod
    # def write(save_path: str, filename: str, batch_idx: int,
    #           tensors: Dict[int, torch.Tensor], dtype: str = 'float32') -> None:
    #     """
    #     Write tensors to a memory-mapped file with metadata.

    #     Args:
    #         save_path: Directory to save files
    #         filename: Base name for the memory-mapped file
    #         batch_idx: Batch index for metadata
    #         tensors: Dictionary mapping layer indices to tensors
    #         dtype: NumPy data type to use for storage
    #     """
    #     # Ensure directory exists
    #     os.makedirs(save_path, exist_ok=True)

    #     # Define file paths
    #     mmap_path = os.path.join(save_path, filename)
    #     metadata_path = os.path.join(save_path, f"{os.path.splitext(filename)[0]}_metadata.json")

    #     # Calculate total size needed
    #     total_size = sum(tensor.numel() for tensor in tensors.values())

    #     # Create memory-mapped file
    #     try:
    #         mmap = np.memmap(mmap_path, dtype=np.dtype(dtype), mode="w+", shape=(total_size,))

    #         # Prepare metadata
    #         metadata = {
    #             "batch_idx": batch_idx,
    #             "layers": [],
    #             "offsets": [],
    #             "shapes": [],
    #             "dtype": dtype,
    #             "total_size": total_size
    #         }

    #         # Write tensor data and collect metadata
    #         offset = 0
    #         for layer_idx, tensor in sorted(tensors.items()):
    #             # Convert to NumPy and write to mmap
    #             array = tensor.cpu().detach().numpy().astype(dtype)
    #             size = array.size
    #             shape = array.shape

    #             # Write to memory-mapped file
    #             mmap[offset:offset+size] = array.ravel()

    #             # Update metadata
    #             metadata["layers"].append(int(layer_idx))
    #             metadata["offsets"].append(int(offset))
    #             metadata["shapes"].append(list(shape))

    #             # Update offset for next tensor
    #             offset += size

    #         # Flush to disk
    #         mmap.flush()

    #         # Save metadata
    #         with open(metadata_path, "w") as f:
    #             json.dump(metadata, f, indent=2)

    #         logger.debug(f"Successfully wrote memory-mapped file: {mmap_path}")

    #     except Exception as e:
    #         logger.error(f"Error writing memory-mapped file {mmap_path}: {str(e)}")
    #         raise
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

        # Process tensors and gather metadata
        tensor_metadata = {}
        total_size = 0

        for layer_idx, tensor in tensors.items():
            # Store tensor metadata
            tensor_metadata[layer_idx] = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "size": tensor.numel(),
                "offset": total_size
            }
            total_size += tensor.numel()

        # Create memory-mapped file
        try:
            # Use uint16 as storage type for all tensors (compatible with bfloat16 bit width)
            storage_dtype = 'uint16' if any(tensor.dtype == torch.bfloat16 for tensor in tensors.values()) else dtype
            mmap = np.memmap(mmap_path, dtype=np.dtype(storage_dtype), mode="w+", shape=(total_size,))

            # Prepare general metadata
            metadata = {
                "batch_idx": batch_idx,
                "storage_dtype": storage_dtype,
                "total_size": total_size,
                "tensors": {}
            }

            # Write tensor data and collect metadata
            for layer_idx, tensor in sorted(tensors.items()):
                offset = tensor_metadata[layer_idx]["offset"]
                size = tensor_metadata[layer_idx]["size"]
                tensor_dtype = str(tensor.dtype)

                # Handle different data types
                if tensor.dtype == torch.bfloat16:
                    # For bfloat16, convert to raw bytes representation as uint16
                    # This preserves the exact bit pattern without conversion loss
                    uint16_view = tensor.view(torch.uint16).cpu()
                    mmap[offset:offset+size] = uint16_view.numpy().ravel()
                else:
                    # For other types, use normal numpy conversion
                    array = tensor.cpu().detach().numpy()
                    mmap[offset:offset+size] = array.ravel()

                # Update metadata
                metadata["tensors"][str(layer_idx)] = {
                    "offset": int(offset),
                    "size": int(size),
                    "shape": tensor_metadata[layer_idx]["shape"],
                    "dtype": tensor_dtype
                }

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

    # @staticmethod
    # def load_tensor_dict(path: str, filename: str) -> Dict[int, torch.Tensor]:
    #     """
    #     Load tensor dictionary from memory-mapped file and metadata.

    #     Args:
    #         path: Directory containing the files
    #         filename: Base name of the memory-mapped file

    #     Returns:
    #         Dictionary mapping layer indices to tensors
    #     """
    #     # Define file paths
    #     mmap_path = os.path.join(path, filename)
    #     metadata_path = os.path.join(path, f"{os.path.splitext(filename)[0]}_metadata.json")

    #     # Check if files exist
    #     if not os.path.exists(mmap_path) or not os.path.exists(metadata_path):
    #         raise FileNotFoundError(f"Missing files: {mmap_path} or {metadata_path}")

    #     # Load metadata
    #     with open(metadata_path, "r") as f:
    #         metadata = json.load(f)

    #     # Get data type
    #     dtype = metadata["dtype"]

    #     # Open memory-mapped file
    #     with MemoryMapHandler.read(path, filename, dtype) as mmap:
    #         # Extract all tensors
    #         result = {}

    #         for i, layer_idx in enumerate(metadata["layers"]):
    #             offset = metadata["offsets"][i]
    #             shape = metadata["shapes"][i]

    #             # Calculate size
    #             size = np.prod(shape)

    #             # Extract array slice and copy to new tensor
    #             array = np.array(mmap[offset:offset+size]).reshape(shape)
    #             tensor = torch.from_numpy(array.copy())

    #             # Store in result dictionary
    #             result[int(layer_idx)] = tensor

    #         return result
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

        # Support both old and new metadata formats
        if "storage_dtype" in metadata:
            # New format with storage_dtype
            storage_dtype = metadata["storage_dtype"]
            tensor_metadata = metadata["tensors"]
        else:
            # Old format
            storage_dtype = metadata["dtype"]
            tensor_metadata = {layer: {"offset": offset, "shape": shape, "dtype": storage_dtype}
                            for layer, offset, shape in zip(
                                metadata["layers"],
                                metadata["offsets"],
                                metadata["shapes"]
                            )}

        # Open memory-mapped file
        with MemoryMapHandler.read(path, filename, storage_dtype) as mmap:
            # Extract all tensors
            result = {}

            for layer_str, tensor_info in tensor_metadata.items():
                layer_idx = int(layer_str)
                offset = tensor_info["offset"]
                shape = tensor_info["shape"]
                dtype_str = tensor_info.get("dtype", storage_dtype)
                size = int(np.prod(shape))

                # Extract array slice
                array_data = np.array(mmap[offset:offset+size])

                # Handle different dtypes
                if dtype_str == "torch.bfloat16":
                    # Convert uint16 back to bfloat16
                    tensor_data = torch.from_numpy(array_data.copy())
                    # Reinterpret the uint16 data as bfloat16
                    tensor = tensor_data.view(torch.bfloat16).reshape(shape)
                else:
                    # Standard numpy to torch conversion
                    tensor_data = torch.from_numpy(array_data.copy())
                    tensor = tensor_data.reshape(shape)

                    # Convert to original dtype if needed
                    if dtype_str != str(tensor.dtype) and hasattr(torch, dtype_str.replace('torch.', '')):
                        target_dtype = getattr(torch, dtype_str.replace('torch.', ''))
                        tensor = tensor.to(target_dtype)

                # Store in result dictionary
                result[layer_idx] = tensor

            return result