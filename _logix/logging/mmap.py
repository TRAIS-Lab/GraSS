# Copyright 2023-present the LogIX team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any, Iterator

import numpy as np
import torch

from _logix.utils import get_logger


class MemoryMapHandler:
    """
    Enhanced MemoryMapHandler for efficient tensor-based storage.
    """

    @staticmethod
    def write_tensor_log(
        save_path: str,
        filename: str,
        data_buffer: List[Tuple[str, Dict[Tuple[str, str], torch.Tensor]]],
        dtype="uint8"
    ) -> None:
        """
        Write tensor logs to a memory-mapped file.

        Args:
            save_path (str): Directory to save the memory-mapped file
            filename (str): Filename for the memory-mapped file
            data_buffer (List[Tuple[str, Dict[Tuple[str, str], torch.Tensor]]]):
                List of (data_id, tensor_dict) tuples
            dtype (str): Data type for the memory-mapped file
        """
        file_root, file_ext = os.path.splitext(filename)
        if file_ext == "":
            filename += ".mmap"

        mmap_filename = os.path.join(save_path, filename)
        metadata_filename = os.path.join(save_path, file_root + "_metadata.json")

        # Calculate total size required for all tensors
        total_size = sum(
            tensor.numel() * tensor.element_size()
            for _, tensor_dict in data_buffer
            for tensor in tensor_dict.values()
        )

        # Create memory-mapped file
        mmap = np.memmap(mmap_filename, dtype=dtype, mode="w+", shape=(total_size,))

        metadata = []
        offset = 0

        # Write each data entry to the memory-mapped file
        for data_id, tensor_dict in data_buffer:
            # Track metadata for this data entry
            entry_metadata = {
                "data_id": data_id,
                "tensors": [],
                "offset": offset
            }

            entry_offset = offset

            # Process each tensor in the tensor_dict
            for (module_name, log_type), tensor in tensor_dict.items():
                # Convert tensor to numpy array for storage
                numpy_tensor = tensor.cpu().detach().numpy()
                tensor_bytes = numpy_tensor.nbytes
                shape = numpy_tensor.shape

                # Store tensor metadata
                tensor_metadata = {
                    "module_name": module_name,
                    "log_type": log_type,
                    "shape": shape,
                    "dtype": str(numpy_tensor.dtype),
                    "offset": offset - entry_offset,
                    "nbytes": tensor_bytes
                }
                entry_metadata["tensors"].append(tensor_metadata)

                # Write tensor data to memory-mapped file
                tensor_bytes = numpy_tensor.tobytes()
                mmap[offset:offset + len(tensor_bytes)] = np.frombuffer(tensor_bytes, dtype=dtype)
                offset += len(tensor_bytes)

            # Add entry metadata to the overall metadata
            metadata.append(entry_metadata)

        # Ensure data is written to disk
        mmap.flush()
        del mmap  # Release the memmap object

        # Write metadata to JSON file
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def write(
        save_path: str,
        filename: str,
        data_buffer: List[Tuple[str, Any]],
        path: List[Tuple[str, str]],
        dtype="uint8"
    ) -> None:
        """
        Legacy write method for backward compatibility.

        Args:
            save_path (str): Directory to save the memory-mapped file
            filename (str): Filename for the memory-mapped file
            data_buffer (List[Tuple[str, Any]]): List of (data_id, data) tuples
            path (List[Tuple[str, str]]): List of paths used for metadata
            dtype (str): Data type for the memory-mapped file
        """
        file_root, file_ext = os.path.splitext(filename)
        if file_ext == "":
            filename += ".mmap"

        mmap_filename = os.path.join(save_path, filename)
        metadata_filename = os.path.join(save_path, file_root + "_metadata.json")

        # Calculate total bytes needed
        total_size = 0
        for _, nested_dict in data_buffer:
            for module_name, log_dict in nested_dict.items():
                for log_type, tensor in log_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        total_size += tensor.numel() * tensor.element_size()
                    elif isinstance(tensor, np.ndarray):
                        total_size += tensor.nbytes

        # Create memory-mapped file
        mmap = np.memmap(mmap_filename, dtype=dtype, mode="w+", shape=(total_size,))

        metadata = []
        offset = 0

        # Process each data entry
        for data_id, nested_dict in data_buffer:
            data_entry = {
                "data_id": data_id,
                "path": [],
                "shape": [],
                "offset": offset,
                "block_size": 0,
                "dtype": None
            }

            block_size = 0
            for module_path in path:
                module_name, log_type = module_path
                # Navigate through nested dictionary
                if module_name not in nested_dict or log_type not in nested_dict[module_name]:
                    get_logger().warning(f"Path {module_name}.{log_type} not found for data_id {data_id}")
                    continue

                tensor = nested_dict[module_name][log_type]
                data_entry["path"].append([module_name, log_type])

                if isinstance(tensor, torch.Tensor):
                    numpy_array = tensor.cpu().detach().numpy()
                else:
                    numpy_array = tensor

                data_entry["shape"].append(numpy_array.shape)
                data_entry["dtype"] = str(numpy_array.dtype)

                # Add tensor size to block size
                tensor_size = np.prod(numpy_array.shape)
                block_size += tensor_size

                # Write tensor data to memory-mapped file
                bytes_data = numpy_array.tobytes()
                mmap[offset:offset + len(bytes_data)] = np.frombuffer(bytes_data, dtype=dtype)
                offset += len(bytes_data)

            data_entry["block_size"] = block_size
            metadata.append(data_entry)

        # Ensure data is written to disk
        mmap.flush()
        del mmap

        # Write metadata to JSON file
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    @contextmanager
    def read(path: str, filename: str, dtype="uint8") -> Iterator[np.memmap]:
        """
        Read a memory-mapped file.

        Args:
            path (str): Directory containing the memory-mapped file
            filename (str): Filename of the memory-mapped file
            dtype (str): Data type of the memory-mapped file

        Yields:
            np.memmap: Memory-mapped array
        """
        _, file_ext = os.path.splitext(filename)
        if file_ext == "":
            filename += ".mmap"

        mmap = np.memmap(os.path.join(path, filename), dtype=dtype, mode="r+")
        try:
            yield mmap
        finally:
            del mmap

    @staticmethod
    def read_metafile(path: str, meta_filename: str) -> List[Dict[str, Any]]:
        """
        Read metadata file.

        Args:
            path (str): Directory containing the metadata file
            meta_filename (str): Filename of the metadata file

        Returns:
            List[Dict[str, Any]]: Metadata as a list of dictionaries
        """
        _, file_ext = os.path.splitext(meta_filename)
        if file_ext == "":
            meta_filename += ".json"
        with open(os.path.join(path, meta_filename), "r") as f:
            metadata = json.load(f)  # This throws error when file does not exist.
        return metadata