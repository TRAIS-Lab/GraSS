"""
Enhanced Memory-mapped file operations with tensor-based storage for efficient gradient handling.
"""

import json
import os
import logging
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

class ChunkedMemoryMapHandler:
    """
    Enhanced handler for memory-mapped file operations with tensor-based chunking.
    Stores gradients as concatenated tensors for better performance.
    """

    @staticmethod
    def write_chunk(save_path: str, data_type: str, chunk_data: List[Tuple[int, Dict[int, torch.Tensor]]],
                   layer_dims: List[int], dtype: str = 'float32') -> str:
        """
        Write multiple batches to a single chunked memory-mapped file as a concatenated tensor.

        Args:
            save_path: Directory to save files
            data_type: Type of data being stored (gradients, ifvp, etc.)
            chunk_data: List of (batch_idx, tensor_dict) tuples
            layer_dims: List of projection dimensions for each layer
            dtype: NumPy data type to use for storage

        Returns:
            The generated chunk filename (without extension)
        """
        os.makedirs(save_path, exist_ok=True)

        # Generate filename based on batch range
        batch_indices = [batch_idx for batch_idx, _ in chunk_data]
        batch_start = min(batch_indices)
        batch_end = max(batch_indices)

        chunk_filename = f"chunk_{data_type}_{batch_start}_{batch_end}"
        mmap_path = os.path.join(save_path, f"{chunk_filename}.mmap")
        metadata_path = os.path.join(save_path, f"{chunk_filename}_metadata.json")

        # Process all batches and create concatenated tensor
        all_batch_tensors = []
        batch_metadata = []
        total_proj_dim = sum(layer_dims)
        num_layers = len(layer_dims)

        for batch_idx, tensor_dict in chunk_data:
            # Concatenate all layers for this batch
            batch_layers = []
            for layer_idx in range(num_layers):
                if layer_idx in tensor_dict and tensor_dict[layer_idx].numel() > 0:
                    batch_layers.append(tensor_dict[layer_idx])
                else:
                    # Create zero tensor with appropriate shape
                    batch_size = batch_layers[0].shape[0] if batch_layers else 1
                    zero_tensor = torch.zeros(batch_size, layer_dims[layer_idx])
                    batch_layers.append(zero_tensor)

            # Concatenate along feature dimension
            if batch_layers:
                batch_tensor = torch.cat(batch_layers, dim=1)  # Shape: (batch_size, total_proj_dim)
                all_batch_tensors.append(batch_tensor)

                batch_metadata.append({
                    "batch_idx": batch_idx,
                    "batch_size": batch_tensor.shape[0],
                    "offset": len(all_batch_tensors) - 1
                })

        if not all_batch_tensors:
            raise ValueError("No valid data to write")

        # Concatenate all batches
        full_tensor = torch.cat(all_batch_tensors, dim=0)  # Shape: (total_samples, total_proj_dim)
        total_samples, total_features = full_tensor.shape

        # Create memory-mapped file
        try:
            # Determine storage dtype
            storage_dtype = 'uint16' if full_tensor.dtype == torch.bfloat16 else dtype

            mmap = np.memmap(mmap_path, dtype=np.dtype(storage_dtype), mode="w+",
                            shape=(total_samples, total_features))

            # Write tensor data
            if full_tensor.dtype == torch.bfloat16:
                uint16_view = full_tensor.view(torch.uint16).cpu()
                mmap[:] = uint16_view.numpy()
            else:
                mmap[:] = full_tensor.cpu().numpy()

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
                "shape": [total_samples, total_features],
                "total_proj_dim": total_proj_dim,
                "layer_dims": layer_dims,
                "num_layers": num_layers,
                "batch_count": len(chunk_data),
                "batches": batch_metadata
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Wrote tensor chunk {chunk_filename}: shape={full_tensor.shape}")

            return chunk_filename

        except Exception as e:
            logger.error(f"Error writing chunked memory-mapped file {mmap_path}: {str(e)}")
            raise

    @staticmethod
    @contextmanager
    def read_chunk(path: str, chunk_filename: str, metadata: Dict[str, Any]):
        """
        Context manager to read from a chunked memory-mapped file.

        Args:
            path: Directory containing the memory-mapped file
            chunk_filename: Name of the chunk file
            metadata: Metadata dictionary containing storage info

        Yields:
            memory-mapped array object with shape (total_samples, total_proj_dim)
        """
        mmap_path = os.path.join(path, f"{chunk_filename}.mmap")
        mmap = None
        try:
            storage_dtype = metadata["storage_dtype"]
            shape = tuple(metadata["shape"])
            mmap = np.memmap(mmap_path, dtype=np.dtype(storage_dtype), mode="r", shape=shape)
            yield mmap
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
    def load_chunk_batch_dict(path: str, chunk_filename: str, batch_idx: int) -> Dict[int, torch.Tensor]:
        """
        Load a specific batch from a chunked memory-mapped file and split into layers.

        Args:
            path: Directory containing the files
            chunk_filename: Name of the chunk file
            batch_idx: Batch index to load

        Returns:
            Dictionary mapping layer indices to tensors
        """
        metadata = ChunkedMemoryMapHandler.read_chunk_metadata(path, chunk_filename)

        # Find the target batch
        target_batch = None
        for batch_info in metadata["batches"]:
            if batch_info["batch_idx"] == batch_idx:
                target_batch = batch_info
                break

        if target_batch is None:
            raise ValueError(f"Batch {batch_idx} not found in chunk {chunk_filename}")

        layer_dims = metadata["layer_dims"]
        num_layers = metadata["num_layers"]

        with ChunkedMemoryMapHandler.read_chunk(path, chunk_filename, metadata) as mmap:
            # Calculate row indices for this batch
            batch_offset = target_batch["offset"]
            batch_size = target_batch["batch_size"]

            # Calculate actual row indices
            row_start = sum(b["batch_size"] for b in metadata["batches"][:batch_offset])
            row_end = row_start + batch_size

            # Extract batch data
            batch_data = np.array(mmap[row_start:row_end], copy=True)

            # Convert to torch tensor
            if metadata.get("storage_dtype") == "uint16":
                # Handle bfloat16
                tensor_data = torch.from_numpy(batch_data).view(torch.bfloat16)
            else:
                tensor_data = torch.from_numpy(batch_data)

            # Split into layers
            result = {}
            start_idx = 0
            for layer_idx, layer_dim in enumerate(layer_dims):
                end_idx = start_idx + layer_dim
                result[layer_idx] = tensor_data[:, start_idx:end_idx].contiguous()
                start_idx = end_idx

        return result

    @staticmethod
    def load_chunk_all_batches(path: str, chunk_filename: str) -> Dict[int, Dict[int, torch.Tensor]]:
        """
        Load all batches from a chunked memory-mapped file.

        Args:
            path: Directory containing the files
            chunk_filename: Name of the chunk file

        Returns:
            Dictionary mapping batch indices to tensor dictionaries
        """
        metadata = ChunkedMemoryMapHandler.read_chunk_metadata(path, chunk_filename)
        layer_dims = metadata["layer_dims"]

        result = {}

        with ChunkedMemoryMapHandler.read_chunk(path, chunk_filename, metadata) as mmap:
            # Convert entire array to tensor
            if metadata.get("storage_dtype") == "uint16":
                full_data = torch.from_numpy(np.array(mmap, copy=True)).view(torch.bfloat16)
            else:
                full_data = torch.from_numpy(np.array(mmap, copy=True))

            # Process each batch
            row_offset = 0
            for batch_info in metadata["batches"]:
                batch_idx = batch_info["batch_idx"]
                batch_size = batch_info["batch_size"]

                # Extract batch rows
                batch_data = full_data[row_offset:row_offset + batch_size]

                # Split into layers
                batch_dict = {}
                start_idx = 0
                for layer_idx, layer_dim in enumerate(layer_dims):
                    end_idx = start_idx + layer_dim
                    batch_dict[layer_idx] = batch_data[:, start_idx:end_idx].contiguous()
                    start_idx = end_idx

                result[batch_idx] = batch_dict
                row_offset += batch_size

        return result

    @staticmethod
    def load_chunk_as_tensor(path: str, chunk_filename: str, batch_range: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, Dict[int, Tuple[int, int]]]:
        """
        Load chunk as a single concatenated tensor with batch mapping.

        Args:
            path: Directory containing the files
            chunk_filename: Name of the chunk file
            batch_range: Optional (start, end) range to filter batches

        Returns:
            Tuple of:
                - Concatenated tensor of shape (total_samples, total_proj_dim)
                - Mapping from batch_idx to (start_row, end_row) in the tensor
        """
        metadata = ChunkedMemoryMapHandler.read_chunk_metadata(path, chunk_filename)

        with ChunkedMemoryMapHandler.read_chunk(path, chunk_filename, metadata) as mmap:
            # Convert to tensor
            if metadata.get("storage_dtype") == "uint16":
                full_tensor = torch.from_numpy(np.array(mmap, copy=True)).view(torch.bfloat16)
            else:
                full_tensor = torch.from_numpy(np.array(mmap, copy=True))

            # Build batch mapping
            batch_mapping = {}
            row_offset = 0

            for batch_info in metadata["batches"]:
                batch_idx = batch_info["batch_idx"]
                batch_size = batch_info["batch_size"]

                # Apply batch range filter if specified
                if batch_range is not None:
                    start_batch, end_batch = batch_range
                    if not (start_batch <= batch_idx < end_batch):
                        row_offset += batch_size
                        continue

                batch_mapping[batch_idx] = (row_offset, row_offset + batch_size)
                row_offset += batch_size

            # If filtering, extract only relevant rows
            if batch_range is not None and len(batch_mapping) < len(metadata["batches"]):
                valid_rows = []
                for start_row, end_row in batch_mapping.values():
                    valid_rows.extend(range(start_row, end_row))

                if valid_rows:
                    filtered_tensor = full_tensor[valid_rows]

                    # Rebuild mapping for filtered tensor
                    new_mapping = {}
                    new_offset = 0
                    for batch_idx, (old_start, old_end) in sorted(batch_mapping.items()):
                        batch_size = old_end - old_start
                        new_mapping[batch_idx] = (new_offset, new_offset + batch_size)
                        new_offset += batch_size

                    return filtered_tensor, new_mapping
                else:
                    return torch.empty(0, metadata["total_proj_dim"]), {}

            return full_tensor, batch_mapping

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