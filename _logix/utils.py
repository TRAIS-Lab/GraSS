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

import hashlib
import logging as default_logging
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from _logix.tensor_log import TensorLog

_logger = None


class DistributedRankFilter(default_logging.Filter):
    """
    This is a logging filter which will filter out logs from all ranks
    in distributed training except for rank 0.
    """

    def filter(self, record):
        return get_rank() == 0


def get_logger() -> default_logging.Logger:
    """
    Get global logger.
    """
    global _logger
    if _logger:
        return _logger
    logger = default_logging.getLogger("AnaLog")
    log_format = default_logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    logger.propagate = False
    logger.setLevel(default_logging.INFO)
    ch = default_logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(default_logging.INFO)
    ch.setFormatter(log_format)

    # Apply the rank filter to the handler
    rank_filter = DistributedRankFilter()
    ch.addFilter(rank_filter)

    logger.addHandler(ch)

    _logger = logger
    return _logger


def to_numpy(tensor) -> np.ndarray:
    """
    Convert a tensor to NumPy array.

    Args:
        tensor: The tensor to be converted.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    else:
        raise ValueError("Unsupported tensor type. Supported libraries: NumPy, PyTorch")


def get_world_size() -> int:
    """
    Get the number of processes in the current distributed group.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 0


def get_rank(group=None) -> int:
    """
    Get the rank of the current process in the current distributed group.

    Args:
        group (optional): The process group to work on. If not specified,
            the default process group will be used.
    """
    if dist.is_initialized():
        return dist.get_rank(group)
    else:
        return 0


def get_repr_dim(named_modules) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Get the representation dimensions and paths for all named modules.

    Args:
        named_modules: Dictionary mapping module objects to their names

    Returns:
        Tuple containing a list of paths and a list of dimensions
    """
    repr_dims = []
    paths = []
    for k, v in named_modules.items():
        get_logger().info(f"{v}: {k}")
        repr_dims.append(k.weight.data.numel())
        paths.append((v, "grad"))  # hardcoded
    return paths, repr_dims


def print_tracked_modules(repr_dim) -> None:
    """
    Print the tracked modules.
    """
    get_logger().info("Tracking the following modules:")
    get_logger().info(f"Total number of parameters: {repr_dim:,}\n")


def module_check(
    module: nn.Module,
    module_name: str,
    supported_modules: Optional[List[nn.Module]] = None,
    type_filter: Optional[List[nn.Module]] = None,
    name_filter: Optional[List[str]] = None,
    is_lora: bool = False,
) -> bool:
    """
    Check if the module is supported for logging.

    Args:
        module (nn.Module): The module to check.
        module_name (str): Name of the module.
        supported_modules (Optional[List[nn.Module]]): List of supported module types.
        type_filter (Optional[List[nn.Module]]): List of module types to filter.
        name_filter (Optional[List[str]]): List of keywords to filter module names.
        is_lora (bool): Flag to check for specific 'analog_lora_B' in module names.

    Returns:
        bool: True if module is supported, False otherwise.
    """
    if list(module.children()):
        return False
    if supported_modules and not isinstance(module, tuple(supported_modules)):
        return False
    if type_filter and not isinstance(module, tuple(type_filter)):
        return False
    if name_filter and not any(keyword in module_name for keyword in name_filter):
        return False
    if is_lora and "logix_lora_B" not in module_name:
        return False
    return True


# Legacy function kept for backward compatibility
def nested_dict():
    """
    Helper function to create a nested defaultdict.
    """
    return defaultdict(nested_dict)


def flatten_tensor_log(tensor_log: TensorLog, paths: List[Tuple[str, str]]) -> torch.Tensor:
    """
    Flatten a TensorLog into a single tensor based on the given paths.

    Args:
        tensor_log (TensorLog): The tensor log to flatten
        paths (List[Tuple[str, str]]): List of (module_name, log_type) paths

    Returns:
        torch.Tensor: Flattened tensor
    """
    flat_tensors = []
    # Extract batch size from the first tensor
    first_tensor = tensor_log.get(paths[0][0], paths[0][1])
    if first_tensor is None:
        raise ValueError(f"Path {paths[0]} not found in tensor log")
    bsz = first_tensor.shape[0]

    # Collect and flatten all tensors
    for module_name, log_type in paths:
        tensor = tensor_log.get(module_name, log_type)
        if tensor is None:
            raise ValueError(f"Path ({module_name}, {log_type}) not found in tensor log")
        flat_tensors.append(tensor.reshape(bsz, -1))

    # Concatenate along the second dimension
    return torch.cat(flat_tensors, dim=1)


def synchronize_tensors(tensors: List[torch.Tensor], device: Optional[torch.device] = None) -> List[torch.Tensor]:
    """
    Move a list of tensors to the specified device.

    Args:
        tensors (List[torch.Tensor]): List of tensors to synchronize
        device (Optional[torch.device]): Target device. If None, uses the device of the first tensor.

    Returns:
        List[torch.Tensor]: List of tensors on the target device
    """
    if not tensors:
        return []

    target_device = device if device is not None else tensors[0].device
    return [tensor.to(device=target_device) for tensor in tensors]


def synchronize_tensor_log(tensor_log: TensorLog, device: Optional[torch.device] = None) -> TensorLog:
    """
    Move all tensors in a TensorLog to the specified device.

    Args:
        tensor_log (TensorLog): The tensor log to synchronize
        device (Optional[torch.device]): Target device. If None, uses the device of the first tensor.

    Returns:
        TensorLog: The synchronized tensor log
    """
    # Return a new TensorLog with all tensors moved to the target device
    return tensor_log.to(device)


def cat_tensor_logs(logs: List[TensorLog], dim: int = 0) -> TensorLog:
    """
    Concatenate multiple TensorLogs along the specified dimension.

    Args:
        logs (List[TensorLog]): List of TensorLogs to concatenate
        dim (int): Dimension along which to concatenate

    Returns:
        TensorLog: Concatenated TensorLog
    """
    if not logs:
        return TensorLog()

    result = logs[0].clone()
    for log in logs[1:]:
        result.cat(log, dim=dim)

    return result


# For backward compatibility with existing code
def flatten_log(log, path) -> torch.Tensor:
    """
    Legacy function to flatten nested dictionary logs. Now converts to TensorLog first.

    Args:
        log: The old-style nested dictionary log
        path: List of paths to flatten

    Returns:
        torch.Tensor: Flattened tensor
    """
    # Convert the old-style log to TensorLog if it's not already
    if not isinstance(log, TensorLog):
        tensor_log = TensorLog.from_dict(log)
    else:
        tensor_log = log

    return flatten_tensor_log(tensor_log, path)


# For backward compatibility with existing code
def synchronize_device(
    src: Dict[str, Dict[str, torch.Tensor]],
    tgt: Dict[str, Dict[str, torch.Tensor]],
    device: Optional[torch.device] = None,
) -> None:
    """
    Legacy function to synchronize device of two tensor dicts.

    Args:
        src (Dict[str, Dict[str, torch.Tensor]]): Source tensors
        tgt (Dict[str, Dict[str, torch.Tensor]]): Target tensors
        device (Optional[torch.device]): Device to synchronize to
    """
    for module_name, module_dict in tgt.items():
        for log in module_dict.keys():
            if device is None and module_name in src and log in src[module_name]:
                device = src[module_name][log].device
            tgt[module_name][log] = tgt[module_name][log].to(device=device)

def merge_logs(log_list):
    """
    Legacy function for merging multiple logs.

    Args:
        log_list: List of (data_id, log_dict) tuples

    Returns:
        Tuple: (merged_data_id, merged_log_dict)
    """
    from _logix.tensor_log import TensorLog

    merged_data_ids = []
    merged_tensor_log = TensorLog()

    for data_id, log_dict in log_list:
        merged_data_ids.extend(data_id)

        # Convert to TensorLog if needed
        if not isinstance(log_dict, TensorLog):
            log = TensorLog.from_dict(log_dict)
        else:
            log = log_dict

        # Merge with accumulated log
        merged_tensor_log.cat(log)

    # For backward compatibility, convert back to nested dict
    return merged_data_ids, merged_tensor_log.to_dict()

class DataIDGenerator:
    """
    Generate unique IDs for data.
    """

    def __init__(self, mode="hash") -> None:
        self.mode = mode
        if mode == "index":
            self.count = 0

    def __call__(self, data: Any):
        if self.mode == "hash":
            return self.generate_hash_id(data)
        elif self.mode == "index":
            return self.generate_index_id(data)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def generate_hash_id(self, data: Any) -> List[str]:
        """
        Given data, generate id using SHA256 hash.
        """
        data_id = []
        for d in data:
            ndarray = to_numpy(d)
            ndarray.flags.writeable = False
            data_id.append(hashlib.sha256(ndarray.tobytes()).hexdigest())
        return data_id

    def generate_index_id(self, data: Any) -> List[str]:
        """
        Given data, generate id based on the index.
        """
        data_id = np.arange(self.count, self.count + len(data))
        data_id = [str(d) for d in data_id]
        self.count += len(data)
        return data_id