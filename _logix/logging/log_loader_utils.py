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

import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import default_collate

from _logix.logging.mmap import MemoryMapHandler
from _logix.tensor_log import TensorLog


def extract_rank_and_chunk(filename):
    """
    Extracts the rank and chunk index from the filename.

    Args:
        filename (str): Filename to extract rank and chunk index from.

    Returns:
        Tuple[int, int]: Tuple containing the rank and chunk index.
    """
    match = re.search(r"rank_(\d+)_chunk_(\d+)", filename)
    return int(match.group(1)), int(match.group(2))


def find_chunk_indices(path) -> List:
    """
    Finds and returns the sorted list of chunk indices based on the filenames in the input path.

    Args:
        path (str): The path to search for chunk files.

    Returns:
        List[int]: Sorted list of chunk indices.
    """
    chunk_indices = []
    for filename in os.listdir(path):
        if filename.endswith(".mmap") and filename.startswith("log_"):
            chunk_index = filename.rstrip(".mmap").strip("log_")
            chunk_indices.append(chunk_index)

    return sorted(chunk_indices, key=extract_rank_and_chunk)


def get_mmap_data(path, mmap_filename, dtype="uint8") -> List:
    """
    Adds memory-mapped files for the given mmap file.

    Args:
        path (str): Path to the directory containing the mmap file.
        mmap_filename (str): Filename of the mmap file.

    Returns:
       List: A list of memory maps.
    """
    with MemoryMapHandler.read(path, mmap_filename, dtype) as mm:
        return mm


def get_mmap_metadata(
    data_id_to_chunk, path, metadata_filename, chunk_index
) -> Dict:
    """
    Reads metadata from a file and updates data_id_to_chunk mapping.

    Args:
        data_id_to_chunk (Dict): Mapping from data_id to chunk info
        path (str): Path to metadata file
        metadata_filename (str): Name of metadata file
        chunk_index: Index of the chunk

    Returns:
        Dict: Updated data_id_to_chunk mapping
    """
    metadata = MemoryMapHandler.read_metafile(path, metadata_filename)
    # Update the mapping from data_id to chunk
    for entry in metadata:
        if entry["data_id"] in data_id_to_chunk:
            logging.warning(f"duplicated data_id detected: {entry['data_id']}")
            continue
        data_id_to_chunk[entry["data_id"]] = (chunk_index, entry)
    return data_id_to_chunk


def collate_tensor_logs(batch):
    """
    Collate a batch of (data_id, TensorLog) pairs.

    Args:
        batch: List of (data_id, TensorLog) pairs

    Returns:
        Tuple[List, TensorLog]: Collated batch as (data_ids, combined_tensor_log)
    """
    # Extract data_ids and tensor_logs
    data_ids = [data_id for data_id, _ in batch]
    tensor_logs = [tensor_log for _, tensor_log in batch]

    # Create a new TensorLog to hold the combined data
    combined_log = TensorLog()

    # Get all unique (module_name, log_type) pairs across all tensor_logs
    all_keys = set()
    for tensor_log in tensor_logs:
        all_keys.update(tensor_log._tensor_store.keys())

    # For each (module_name, log_type) pair, concatenate the tensors from all logs
    for key in all_keys:
        module_name, log_type = key
        # Collect tensors for this key from all logs
        tensors = []
        for tensor_log in tensor_logs:
            tensor = tensor_log.get(module_name, log_type)
            if tensor is not None:
                tensors.append(tensor)

        if tensors:
            # Concatenate tensors along the batch dimension (0)
            combined_tensor = torch.cat(tensors, dim=0)
            combined_log.add(module_name, log_type, combined_tensor)

    return data_ids, combined_log


def collate_tensor_dicts(batch):
    """
    Collate a batch of (data_id, tensor_dict) pairs for legacy support.

    Args:
        batch: List of (data_id, tensor_dict) pairs

    Returns:
        Tuple[List, Dict]: Collated batch as (data_ids, combined_tensor_dict)
    """
    # Extract data_ids and tensor_dicts
    data_ids = [data_id for data_id, _ in batch]
    tensor_dicts = [tensor_dict for _, tensor_dict in batch]

    # Create a new dictionary to hold the combined data
    combined_dict = {}

    # Get all unique module_name, log_type pairs across all tensor_dicts
    all_keys = set()
    for tensor_dict in tensor_dicts:
        all_keys.update(tensor_dict.keys())

    # For each (module_name, log_type) pair, concatenate the tensors from all dicts
    for key in all_keys:
        # Collect tensors for this key from all dicts
        tensors = []
        for tensor_dict in tensor_dicts:
            if key in tensor_dict:
                tensors.append(tensor_dict[key])

        if tensors:
            # Concatenate tensors along the batch dimension (0)
            combined_tensor = torch.cat(tensors, dim=0)
            combined_dict[key] = combined_tensor

    return data_ids, combined_dict


def get_tensor_from_mmap(mmap, offset, shape, dtype_str):
    """
    Extract a tensor from a memory-mapped file.

    Args:
        mmap: Memory-mapped file
        offset (int): Offset in the memory-mapped file
        shape (tuple): Shape of the tensor
        dtype_str (str): Data type of the tensor

    Returns:
        torch.Tensor: Extracted tensor
    """
    # Get numpy dtype from dtype string
    dtype = np.dtype(dtype_str)

    # Calculate size in bytes
    size_in_bytes = np.prod(shape) * dtype.itemsize

    # Extract array from memory-mapped file
    array = np.ndarray(
        shape,
        dtype,
        buffer=mmap,
        offset=offset,
        order='C'
    )

    # Convert to torch tensor
    return torch.from_numpy(array.copy())


# For backward compatibility
def collate_nested_dicts(batch):
    """
    Legacy collation function for nested dictionaries.

    Args:
        batch: List of (data_id, nested_dict) pairs

    Returns:
        Tuple: Collated data
    """
    # Convert nested dicts to TensorLogs
    tensor_log_batch = []
    for data_id, nested_dict in batch:
        tensor_log = TensorLog.from_dict(nested_dict)
        tensor_log_batch.append((data_id, tensor_log))

    # Use tensor log collation
    return collate_tensor_logs(tensor_log_batch)