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

from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
import torch


class TensorLog:
    """
    TensorLog is a more efficient tensor-based replacement for the nested dictionary structure.
    It stores tensors directly in a flat structure indexed by module name and log type.
    """

    def __init__(self):
        # Main storage: mapping from (module_name, log_type) to tensor
        self._tensor_store: Dict[Tuple[str, str], torch.Tensor] = {}
        # Track all available module names
        self._module_names: List[str] = []
        # Track all available log types
        self._log_types: List[str] = []

    def add(self, module_name: str, log_type: str, tensor: torch.Tensor) -> None:
        """
        Add a tensor to the log under the given module name and log type.

        Args:
            module_name (str): Name of the module (e.g., "layer1.linear")
            log_type (str): Type of log (e.g., "forward", "backward", "grad")
            tensor (torch.Tensor): Tensor to store
        """
        key = (module_name, log_type)

        # Update our tracking lists if this is a new module or log type
        if module_name not in self._module_names:
            self._module_names.append(module_name)
        if log_type not in self._log_types:
            self._log_types.append(log_type)

        # Store or add to the tensor
        if key in self._tensor_store:
            self._tensor_store[key] += tensor
        else:
            self._tensor_store[key] = tensor

    def get(self, module_name: str, log_type: str) -> Optional[torch.Tensor]:
        """
        Get the tensor for the given module name and log type.

        Args:
            module_name (str): Name of the module
            log_type (str): Type of log

        Returns:
            Optional[torch.Tensor]: The stored tensor or None if not found
        """
        key = (module_name, log_type)
        return self._tensor_store.get(key)

    def get_all_modules(self) -> List[str]:
        """
        Get all module names in the log.

        Returns:
            List[str]: List of all module names
        """
        return self._module_names

    def get_all_log_types(self) -> List[str]:
        """
        Get all log types in the log.

        Returns:
            List[str]: List of all log types
        """
        return self._log_types

    def items(self) -> Iterator[Tuple[Tuple[str, str], torch.Tensor]]:
        """
        Iterate over all items in the log.

        Returns:
            Iterator: Iterator over (module_name, log_type), tensor pairs
        """
        return self._tensor_store.items()

    def module_items(self, module_name: str) -> Dict[str, torch.Tensor]:
        """
        Get all log items for a specific module.

        Args:
            module_name (str): Name of the module

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping log_type to tensor
        """
        return {log_type: tensor
                for (mod_name, log_type), tensor in self._tensor_store.items()
                if mod_name == module_name}

    def log_type_items(self, log_type: str) -> Dict[str, torch.Tensor]:
        """
        Get all module items for a specific log type.

        Args:
            log_type (str): Type of log

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping module_name to tensor
        """
        return {module_name: tensor
                for (module_name, log_tp), tensor in self._tensor_store.items()
                if log_tp == log_type}

    def clear(self) -> None:
        """
        Clear all tensors in the log.
        """
        self._tensor_store.clear()
        # Don't clear module names and log types as they might be needed for metadata

    def to_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Convert to the old nested dictionary format for backward compatibility.

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: Nested dictionary form of the log
        """
        result = {}
        for (module_name, log_type), tensor in self._tensor_store.items():
            if module_name not in result:
                result[module_name] = {}
            result[module_name][log_type] = tensor
        return result

    @classmethod
    def from_dict(cls, nested_dict: Dict[str, Dict[str, Any]]) -> 'TensorLog':
        """
        Create a TensorLog from a nested dictionary.

        Args:
            nested_dict: Nested dictionary mapping module_name -> log_type -> tensor

        Returns:
            TensorLog: New TensorLog instance with the data from the dictionary
        """
        tensor_log = cls()
        for module_name, log_dict in nested_dict.items():
            for log_type, tensor in log_dict.items():
                if isinstance(tensor, torch.Tensor):
                    tensor_log.add(module_name, log_type, tensor)
        return tensor_log

    def __len__(self) -> int:
        """
        Get the number of tensors in the log.

        Returns:
            int: Number of tensors
        """
        return len(self._tensor_store)

    def __contains__(self, key: Tuple[str, str]) -> bool:
        """
        Check if a module_name, log_type pair exists in the log.

        Args:
            key: Tuple of (module_name, log_type)

        Returns:
            bool: True if the key exists
        """
        return key in self._tensor_store

    def __getitem__(self, key: Tuple[str, str]) -> torch.Tensor:
        """
        Get a tensor by (module_name, log_type) key.

        Args:
            key: Tuple of (module_name, log_type)

        Returns:
            torch.Tensor: The stored tensor

        Raises:
            KeyError: If the key does not exist
        """
        return self._tensor_store[key]

    def clone(self) -> 'TensorLog':
        """
        Create a deep copy of this TensorLog.

        Returns:
            TensorLog: A new TensorLog with cloned tensors
        """
        new_log = TensorLog()
        new_log._module_names = self._module_names.copy()
        new_log._log_types = self._log_types.copy()
        new_log._tensor_store = {k: v.clone() for k, v in self._tensor_store.items()}
        return new_log

    def cat(self, other: 'TensorLog', dim: int = 0) -> None:
        """
        Concatenate another TensorLog to this one along the specified dimension.
        This is useful for combining logs from different batches.

        Args:
            other (TensorLog): The other TensorLog to concatenate
            dim (int): Dimension along which to concatenate
        """
        for (module_name, log_type), tensor in other._tensor_store.items():
            if (module_name, log_type) in self._tensor_store:
                self._tensor_store[(module_name, log_type)] = torch.cat(
                    [self._tensor_store[(module_name, log_type)], tensor], dim=dim
                )
            else:
                self.add(module_name, log_type, tensor)

                # Add to tracking lists if needed
                if module_name not in self._module_names:
                    self._module_names.append(module_name)
                if log_type not in self._log_types:
                    self._log_types.append(log_type)

    def to(self, device: torch.device) -> 'TensorLog':
        """
        Move all tensors to the specified device.

        Args:
            device (torch.device): Device to move tensors to

        Returns:
            TensorLog: Self for chaining
        """
        for key, tensor in self._tensor_store.items():
            self._tensor_store[key] = tensor.to(device)
        return self

    def detach(self) -> 'TensorLog':
        """
        Detach all tensors from the computation graph.

        Returns:
            TensorLog: Self for chaining
        """
        for key, tensor in self._tensor_store.items():
            self._tensor_store[key] = tensor.detach()
        return self