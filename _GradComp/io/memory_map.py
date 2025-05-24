"""
Memory-mapped file operations.
"""

import json
import os
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch

import logging
logger = logging.getLogger(__name__)

class ChunkedMemoryMapHandler:
    """
    Optimized handler for memory-mapped file operations with pure tensor storage.
    """

    @staticmethod
    def write_chunk(
            save_path: str,
            data_type: str,
            tensor: torch.Tensor,
            batch_info: List[Dict[str, Any]],
            layer_dims: List[int],
            dtype: str = 'float32'
        ) -> str:
        """
        Write a tensor chunk directly to memory-mapped file.

        Args:
            save_path: Directory to save files
            data_type: Type of data being stored (gradients, ifvp, etc.)
            tensor: Pre-concatenated tensor of shape (total_samples, total_proj_dim)
            batch_info: List of batch metadata dicts with keys: batch_idx, start_row, end_row
            layer_dims: List of projection dimensions for each layer
            dtype: NumPy data type to use for storage

        Returns:
            The generated chunk filename (without extension)
        """
        os.makedirs(save_path, exist_ok=True)

        # Generate filename based on batch range
        batch_indices = [info["batch_idx"] for info in batch_info]
        batch_start = min(batch_indices)
        batch_end = max(batch_indices)

        chunk_filename = f"chunk_{data_type}_{batch_start}_{batch_end}"
        mmap_path = os.path.join(save_path, f"{chunk_filename}.mmap")
        metadata_path = os.path.join(save_path, f"{chunk_filename}_metadata.json")

        # Create memory-mapped file
        try:
            # Determine storage dtype
            storage_dtype = 'uint16' if tensor.dtype == torch.bfloat16 else dtype

            # Ensure tensor is on CPU
            if tensor.device.type != 'cpu':
                tensor = tensor.cpu()

            # Create memory map with exact tensor shape
            mmap = np.memmap(mmap_path, dtype=np.dtype(storage_dtype), mode="w+",
                            shape=tensor.shape)

            # Write tensor data directly
            if tensor.dtype == torch.bfloat16:
                uint16_view = tensor.view(torch.uint16)
                mmap[:] = uint16_view.numpy()
            else:
                mmap[:] = tensor.numpy()

            # Flush to disk
            mmap.flush()

            # Save minimal metadata
            metadata = {
                "chunk_filename": chunk_filename,
                "data_type": data_type,
                "batch_start": batch_start,
                "batch_end": batch_end,
                "storage_dtype": storage_dtype,
                "shape": list(tensor.shape),
                "layer_dims": layer_dims,
                "batches": batch_info  # Contains batch_idx, start_row, end_row for each batch
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, separators=(',', ':'))  # Compact format

            logger.debug(f"Wrote tensor chunk {chunk_filename}: shape={tensor.shape}")

            return chunk_filename

        except Exception as e:
            logger.error(f"Error writing chunked memory-mapped file {mmap_path}: {str(e)}")
            raise

    @staticmethod
    @contextmanager
    def read_chunk(path: str, chunk_filename: str):
        """
        Context manager to read a chunked memory-mapped file as tensor.

        Args:
            path: Directory containing the memory-mapped file
            chunk_filename: Name of the chunk file

        Yields:
            Tuple of (tensor, metadata) where tensor has shape (total_samples, total_proj_dim)
        """
        mmap_path = os.path.join(path, f"{chunk_filename}.mmap")
        metadata_path = os.path.join(path, f"{chunk_filename}_metadata.json")

        mmap = None
        try:
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            storage_dtype = metadata["storage_dtype"]
            shape = tuple(metadata["shape"])

            # Open memory map
            mmap = np.memmap(mmap_path, dtype=np.dtype(storage_dtype), mode="r+", shape=shape)

            # ✅ Create tensor view WITHOUT copying!
            if storage_dtype == "uint16":
                # For bfloat16, we need special handling
                # Create a tensor that shares memory with the mmap
                tensor = torch.as_tensor(mmap, dtype=torch.int16).view(torch.bfloat16)
            else:
                # Directly create tensor view of mmap - no copy!
                tensor = torch.as_tensor(mmap)

            yield tensor, metadata

        finally:
            if mmap is not None:
                del mmap
                import gc
                gc.collect()

    @staticmethod
    def read_chunk_metadata(path: str, chunk_filename: str) -> Dict[str, Any]:
        """Read metadata from a chunked file."""
        metadata_path = os.path.join(path, f"{chunk_filename}_metadata.json")
        with open(metadata_path, "r") as f:
            return json.load(f)

    @staticmethod
    def load_chunk_tensor(path: str, chunk_filename: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Load entire chunk as tensor with metadata.

        Args:
            path: Directory containing the files
            chunk_filename: Name of the chunk file

        Returns:
            Tuple of (tensor, metadata)
        """
        with ChunkedMemoryMapHandler.read_chunk(path, chunk_filename) as (tensor, metadata):
            # Make a copy to ensure it's not tied to the memory map
            return tensor.clone(), metadata

    @staticmethod
    def load_batch_slice(path: str, chunk_filename: str, batch_idx: int) -> Optional[torch.Tensor]:
        """
        Load a specific batch slice from a chunk.

        Args:
            path: Directory containing the files
            chunk_filename: Name of the chunk file
            batch_idx: Batch index to load

        Returns:
            Tensor slice for the batch or None if not found
        """
        with ChunkedMemoryMapHandler.read_chunk(path, chunk_filename) as (tensor, metadata):
            # Find the batch in metadata
            for batch_info in metadata["batches"]:
                if batch_info["batch_idx"] == batch_idx:
                    start_row = batch_info["start_row"]
                    end_row = batch_info["end_row"]
                    return tensor[start_row:end_row].clone()

            return None

    @staticmethod
    def load_chunk_batch_range(
            path: str,
            chunk_filename: str,
            batch_range: Optional[Tuple[int, int]] = None
        ) -> Tuple[torch.Tensor, Dict[int, Tuple[int, int]]]:
        """
        Load chunk with optional batch filtering.

        Args:
            path: Directory containing the files
            chunk_filename: Name of the chunk file
            batch_range: Optional (start, end) range to filter batches

        Returns:
            Tuple of:
                - Tensor of shape (filtered_samples, total_proj_dim)
                - Mapping from batch_idx to (start_row, end_row) in the returned tensor
        """
        with ChunkedMemoryMapHandler.read_chunk(path, chunk_filename) as (tensor, metadata):
            if batch_range is None:
                # ✅ Return tensor directly - no clone needed for read-only access
                batch_mapping = {
                    info["batch_idx"]: (info["start_row"], info["end_row"])
                    for info in metadata["batches"]
                }
                return tensor, batch_mapping

            # Filter batches
            start_batch, end_batch = batch_range
            valid_rows = []
            new_mapping = {}
            new_offset = 0

            for batch_info in metadata["batches"]:
                batch_idx = batch_info["batch_idx"]
                if start_batch <= batch_idx < end_batch:
                    start_row = batch_info["start_row"]
                    end_row = batch_info["end_row"]
                    batch_size = end_row - start_row

                    valid_rows.extend(range(start_row, end_row))
                    new_mapping[batch_idx] = (new_offset, new_offset + batch_size)
                    new_offset += batch_size

            if valid_rows:
                filtered_tensor = tensor[valid_rows].clone()
                return filtered_tensor, new_mapping
            else:
                return torch.empty(0, tensor.shape[1]), {}

    @staticmethod
    def find_chunk_files(path: str, data_type: str) -> List[str]:
        """Find all chunk files for a specific data type."""
        if not os.path.exists(path):
            return []

        chunk_files = []
        for filename in os.listdir(path):
            if filename.endswith("_metadata.json") and f"chunk_{data_type}_" in filename:
                chunk_name = filename.replace("_metadata.json", "")
                chunk_files.append(chunk_name)

        # Sort by batch start index
        def extract_batch_start(chunk_name):
            try:
                parts = chunk_name.split('_')
                if len(parts) >= 4:
                    return int(parts[2])
                return 0
            except (ValueError, IndexError):
                return 0

        return sorted(chunk_files, key=extract_batch_start)