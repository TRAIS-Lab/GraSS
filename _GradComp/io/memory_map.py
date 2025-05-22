"""
Enhanced Memory-mapped file operations with chunking support for efficient tensor storage and access.
"""

# Standard library imports
import json
import os
import logging
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any, Optional

# Third-party imports
import numpy as np
import torch

# Configure logger
logger = logging.getLogger(__name__)

class ChunkedMemoryMapHandler:
    """
    Enhanced handler for memory-mapped file operations with chunking support.
    Uses larger chunks to reduce I/O overhead and file handle count.
    """

    @staticmethod
    def write_chunk(save_path: str, data_type: str, chunk_data: List[Tuple[int, Dict[int, torch.Tensor]]],
                   dtype: str = 'float32') -> str:
        """
        Write multiple batches to a single chunked memory-mapped file.

        Args:
            save_path: Directory to save files
            data_type: Type of data being stored (gradients, ifvp, etc.)
            chunk_data: List of (batch_idx, tensor_dict) tuples
            dtype: NumPy data type to use for storage

        Returns:
            The generated chunk filename (without extension)
        """
        # Ensure directory exists
        os.makedirs(save_path, exist_ok=True)

        # Generate filename based on batch range
        batch_indices = [batch_idx for batch_idx, _ in chunk_data]
        batch_start = min(batch_indices)
        batch_end = max(batch_indices)

        chunk_filename = f"chunk_{data_type}_{batch_start}_{batch_end}"
        mmap_path = os.path.join(save_path, f"{chunk_filename}.mmap")
        metadata_path = os.path.join(save_path, f"{chunk_filename}_metadata.json")

        # Process all batches and gather metadata
        batch_metadata = []
        total_size = 0

        for batch_idx, tensor_dict in chunk_data:
            batch_info = {
                "batch_idx": batch_idx,
                "tensors": {},
                "offset": total_size
            }

            for layer_idx, tensor in tensor_dict.items():
                tensor_size = tensor.numel()
                batch_info["tensors"][str(layer_idx)] = {
                    "offset": total_size,
                    "size": tensor_size,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype)
                }
                total_size += tensor_size

            batch_metadata.append(batch_info)

        # Create memory-mapped file
        try:
            # Use appropriate storage dtype
            storage_dtype = 'uint16' if any(
                any(tensor.dtype == torch.bfloat16 for tensor in batch[1].values())
                for batch in chunk_data
            ) else dtype

            mmap = np.memmap(mmap_path, dtype=np.dtype(storage_dtype), mode="w+", shape=(total_size,))

            # Write all tensor data
            current_offset = 0
            for batch_idx, tensor_dict in chunk_data:
                for layer_idx, tensor in sorted(tensor_dict.items()):
                    size = tensor.numel()

                    # Handle different data types
                    if tensor.dtype == torch.bfloat16:
                        uint16_view = tensor.view(torch.uint16).cpu()
                        mmap[current_offset:current_offset+size] = uint16_view.numpy().ravel()
                    else:
                        array = tensor.cpu().detach().numpy()
                        mmap[current_offset:current_offset+size] = array.ravel()

                    current_offset += size

            # Flush to disk
            mmap.flush()

            # Save metadata
            metadata = {
                "chunk_filename": chunk_filename,
                "data_type": data_type,
                "batch_start": batch_start,
                "batch_end": batch_end,
                "batch_indices": sorted(batch_indices),
                "storage_dtype": storage_dtype,
                "total_size": total_size,
                "batch_count": len(chunk_data),
                "batches": batch_metadata
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Successfully wrote chunked memory-mapped file: {mmap_path}")

            return chunk_filename

        except Exception as e:
            logger.error(f"Error writing chunked memory-mapped file {mmap_path}: {str(e)}")
            raise

    @staticmethod
    @contextmanager
    def read_chunk(path: str, chunk_filename: str, storage_dtype: str = 'float32'):
        """
        Context manager to read from a chunked memory-mapped file.
        Ensures proper cleanup of file handles.

        Args:
            path: Directory containing the memory-mapped file
            chunk_filename: Name of the chunk file
            storage_dtype: NumPy data type used for storage

        Yields:
            memory-mapped array object
        """
        mmap_path = os.path.join(path, f"{chunk_filename}.mmap")
        mmap = None
        try:
            mmap = np.memmap(mmap_path, dtype=np.dtype(storage_dtype), mode="r")
            yield mmap
        finally:
            if mmap is not None:
                # Explicitly delete and flush to close file handle
                del mmap
                # Force garbage collection to ensure cleanup
                import gc
                gc.collect()

    @staticmethod
    def read_chunk_metadata(path: str, chunk_filename: str) -> Dict[str, Any]:
        """
        Read metadata from a chunked file.

        Args:
            path: Directory containing the metadata file
            chunk_filename: Name of the chunk file

        Returns:
            Metadata dictionary
        """
        metadata_path = os.path.join(path, f"{chunk_filename}_metadata.json")
        with open(metadata_path, "r") as f:
            return json.load(f)

    @staticmethod
    def load_chunk_batch_dict(path: str, chunk_filename: str, batch_idx: int) -> Dict[int, torch.Tensor]:
        """
        Load a specific batch from a chunked memory-mapped file.
        Optimized to minimize file handle usage.

        Args:
            path: Directory containing the files
            chunk_filename: Name of the chunk file
            batch_idx: Batch index to load

        Returns:
            Dictionary mapping layer indices to tensors
        """
        # Load metadata once
        metadata = ChunkedMemoryMapHandler.read_chunk_metadata(path, chunk_filename)

        # Find the target batch
        target_batch = None
        for batch_info in metadata["batches"]:
            if batch_info["batch_idx"] == batch_idx:
                target_batch = batch_info
                break

        if target_batch is None:
            raise ValueError(f"Batch {batch_idx} not found in chunk {chunk_filename}")

        # Load data using context manager to ensure cleanup
        storage_dtype = metadata["storage_dtype"]
        result = {}

        with ChunkedMemoryMapHandler.read_chunk(path, chunk_filename, storage_dtype) as mmap:
            for layer_str, tensor_info in target_batch["tensors"].items():
                layer_idx = int(layer_str)
                offset = tensor_info["offset"]
                size = tensor_info["size"]
                shape = tensor_info["shape"]
                dtype_str = tensor_info["dtype"]

                # Extract array slice and immediately copy to avoid keeping mmap reference
                array_data = np.array(mmap[offset:offset+size], copy=True)

                # Handle different dtypes
                if dtype_str == "torch.bfloat16":
                    tensor_data = torch.from_numpy(array_data)
                    tensor = tensor_data.view(torch.bfloat16).reshape(shape)
                else:
                    tensor_data = torch.from_numpy(array_data)
                    tensor = tensor_data.reshape(shape)

                    # Convert to original dtype if needed
                    if dtype_str != str(tensor.dtype) and hasattr(torch, dtype_str.replace('torch.', '')):
                        target_dtype = getattr(torch, dtype_str.replace('torch.', ''))
                        tensor = tensor.to(target_dtype)

                result[layer_idx] = tensor

        return result

    @staticmethod
    def load_chunk_all_batches(path: str, chunk_filename: str) -> Dict[int, Dict[int, torch.Tensor]]:
        """
        Load all batches from a chunked memory-mapped file.
        Optimized to minimize file handle usage and memory overhead.

        Args:
            path: Directory containing the files
            chunk_filename: Name of the chunk file

        Returns:
            Dictionary mapping batch indices to tensor dictionaries
        """
        # Load metadata once
        metadata = ChunkedMemoryMapHandler.read_chunk_metadata(path, chunk_filename)
        storage_dtype = metadata["storage_dtype"]

        result = {}

        # Use context manager to ensure proper cleanup
        with ChunkedMemoryMapHandler.read_chunk(path, chunk_filename, storage_dtype) as mmap:
            for batch_info in metadata["batches"]:
                batch_idx = batch_info["batch_idx"]
                batch_dict = {}

                for layer_str, tensor_info in batch_info["tensors"].items():
                    layer_idx = int(layer_str)
                    offset = tensor_info["offset"]
                    size = tensor_info["size"]
                    shape = tensor_info["shape"]
                    dtype_str = tensor_info["dtype"]

                    # Extract array slice and immediately copy to avoid keeping mmap reference
                    array_data = np.array(mmap[offset:offset+size], copy=True)

                    # Handle different dtypes
                    if dtype_str == "torch.bfloat16":
                        tensor_data = torch.from_numpy(array_data)
                        tensor = tensor_data.view(torch.bfloat16).reshape(shape)
                    else:
                        tensor_data = torch.from_numpy(array_data)
                        tensor = tensor_data.reshape(shape)

                        # Convert to original dtype if needed
                        if dtype_str != str(tensor.dtype) and hasattr(torch, dtype_str.replace('torch.', '')):
                            target_dtype = getattr(torch, dtype_str.replace('torch.', ''))
                            tensor = tensor.to(target_dtype)

                    batch_dict[layer_idx] = tensor

                result[batch_idx] = batch_dict

        return result

    @staticmethod
    def find_chunk_files(path: str, data_type: str) -> List[str]:
        """
        Find all chunk files for a specific data type.

        Args:
            path: Directory to search
            data_type: Type of data ('gradients', 'ifvp', etc.)

        Returns:
            List of chunk filenames (without extensions) sorted by batch range
        """
        if not os.path.exists(path):
            return []

        chunk_files = []
        for filename in os.listdir(path):
            if filename.endswith("_metadata.json") and f"chunk_{data_type}_" in filename:
                # Extract chunk filename without _metadata.json
                chunk_name = filename.replace("_metadata.json", "")
                chunk_files.append(chunk_name)

        # Sort by batch start index for consistent ordering
        def extract_batch_start(chunk_name):
            try:
                # chunk_gradients_0_31 -> extract 0
                parts = chunk_name.split('_')
                if len(parts) >= 4:  # chunk_type_start_end
                    return int(parts[2])  # batch_start
                return 0
            except (ValueError, IndexError):
                return 0

        return sorted(chunk_files, key=extract_batch_start)