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

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset

from _logix.logging.log_loader_utils import (
    find_chunk_indices,
    get_mmap_data,
    get_mmap_metadata,
    get_tensor_from_mmap,
)
from _logix.tensor_log import TensorLog


class LogDataset(Dataset):
    """
    Dataset for loading logged tensor data from memory-mapped files.
    """
    def __init__(self, log_dir, flatten=False):
        """
        Initialize a LogDataset.

        Args:
            log_dir (str): Directory containing the log files
            flatten (bool): Whether to flatten tensors (used for legacy support)
        """
        self.chunk_indices = None
        self.memmaps = []

        self.data_id_to_chunk = OrderedDict()
        self.log_dir = log_dir
        self.flatten = flatten

        # Find all chunk indices
        self.chunk_indices = find_chunk_indices(self.log_dir)
        self.fetch_data()
        self.data_id_list = list(self.data_id_to_chunk.keys())

    def fetch_data(self):
        """
        Fetch metadata and memory-mapped data for all chunks.
        """
        # Add metadata and mmap files for all indices.
        for idx, chunk_index in enumerate(self.chunk_indices):
            file_root = f"log_{chunk_index}"
            mmap_filename = f"{file_root}.mmap"
            entry = get_mmap_data(self.log_dir, mmap_filename)
            self.memmaps.append(entry)

            self.data_id_to_chunk = get_mmap_metadata(
                self.data_id_to_chunk,
                self.log_dir,
                f"{file_root}_metadata.json",
                idx,
            )

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to get

        Returns:
            Tuple: Either (data_id, tensor) or (data_id, TensorLog)
        """
        data_id = self.data_id_list[index]
        chunk_idx, entry = self.data_id_to_chunk[data_id]
        mmap = self.memmaps[chunk_idx]

        # Check if we're using the new tensor log format or the legacy format
        if "tensors" in entry:
            # New tensor log format
            tensor_log = TensorLog()

            # Load each tensor from the metadata
            for tensor_info in entry["tensors"]:
                module_name = tensor_info["module_name"]
                log_type = tensor_info["log_type"]
                shape = tuple(tensor_info["shape"])
                dtype = tensor_info["dtype"]
                offset = entry["offset"] + tensor_info["offset"]

                # Extract tensor from memory map
                tensor = get_tensor_from_mmap(mmap, offset, shape, dtype)

                # Add to tensor log
                tensor_log.add(module_name, log_type, tensor)

            if self.flatten:
                # For backward compatibility, flatten to a single tensor
                # This would need specific path information which should be
                # provided elsewhere in the API
                return data_id, tensor_log
            else:
                return data_id, tensor_log
        else:
            # Legacy format
            nested_dict = {}
            offset = entry["offset"]
            flat_tensor = self._get_flatten_item(
                mmap, offset, entry["block_size"], entry["dtype"]
            )

            if self.flatten:
                return data_id, flat_tensor

            # Unflatten the tensor into the nested dictionary structure
            start = 0
            for i in range(len(entry["path"])):
                path = entry["path"][i]
                shape = tuple(entry["shape"][i])
                tensor, start = self._unflatten_tensor(flat_tensor, shape, start)

                # Navigate through the nested dictionary and place the tensor
                current_level = nested_dict
                for key in path[:-1]:
                    if key not in current_level:
                        current_level[key] = {}
                    current_level = current_level[key]
                current_level[path[-1]] = tensor

            # Convert to TensorLog for consistency
            tensor_log = TensorLog.from_dict(nested_dict)
            return data_id, tensor_log

    def _get_flatten_item(self, mmap, offset, block_size, dtype="float32"):
        """
        Helper method to extract a flattened tensor from memory map.

        Args:
            mmap: Memory map
            offset (int): Offset in the memory map
            block_size (int): Size of the block to extract
            dtype (str): Data type

        Returns:
            torch.Tensor: Extracted tensor
        """
        array = np.ndarray(
            block_size,
            dtype,
            buffer=mmap,
            offset=offset,
            order="C",
        )
        return torch.from_numpy(array.clone())

    def _unflatten_tensor(self, flat_tensor, shape, start):
        """
        Helper method to unflatten a tensor.

        Args:
            flat_tensor (torch.Tensor): Flattened tensor
            shape (tuple): Shape to unflatten to
            start (int): Starting index

        Returns:
            Tuple[torch.Tensor, int]: Unflattened tensor and new starting index
        """
        num_elements = np.prod(shape)
        end = start + num_elements
        unflattened_tensor = flat_tensor[start:end].view(*shape)
        return unflattened_tensor, end

    def __len__(self):
        """
        Get the number of items in the dataset.

        Returns:
            int: Number of items
        """
        return len(self.data_id_to_chunk)

    def close(self):
        """
        Close all memory maps.
        """
        for mmap in self.memmaps:
            del mmap