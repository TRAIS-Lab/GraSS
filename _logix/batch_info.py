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

from typing import Any, Iterable, Optional

import torch

from _logix.tensor_log import TensorLog


class BatchInfo:
    """
    BatchInfo stores information about the current batch during training or evaluation.
    It has been redesigned to use the more efficient TensorLog instead of nested dictionaries.
    """
    def __init__(self):
        """
        Initialize a new BatchInfo instance.
        """
        # A unique identifier or list of identifiers for the current batch
        self.data_id: Optional[Iterable[Any]] = None

        # Optional attention mask or other mask to be applied to activations
        self.mask: Optional[torch.Tensor] = None

        # Log storage using the new efficient TensorLog class
        self.log = TensorLog()

    def clear(self):
        """
        Clear all information stored in this BatchInfo instance.
        """
        self.data_id = None
        self.mask = None
        self.log.clear()

    def copy(self):
        """
        Create a copy of this BatchInfo instance.

        Returns:
            BatchInfo: A new BatchInfo instance with the same data
        """
        new_info = BatchInfo()
        new_info.data_id = self.data_id  # data_id is usually a reference that can be shared

        if self.mask is not None:
            new_info.mask = self.mask.clone()

        # Use the clone method from TensorLog to create a deep copy
        new_info.log = self.log.clone()

        return new_info

    def get_module_logs(self, module_name: str) -> dict:
        """
        Get all logs for a specific module.

        Args:
            module_name (str): The name of the module

        Returns:
            dict: Dictionary mapping log_type to tensor
        """
        return self.log.module_items(module_name)

    def get_log_type(self, log_type: str) -> dict:
        """
        Get logs of a specific type across all modules.

        Args:
            log_type (str): The type of log to retrieve

        Returns:
            dict: Dictionary mapping module_name to tensor
        """
        return self.log.log_type_items(log_type)

    def to_legacy_format(self) -> tuple:
        """
        Convert to the legacy format (data_id, nested_dict) for backward compatibility.

        Returns:
            tuple: (data_id, nested_dict) where nested_dict maps module -> log_type -> tensor
        """
        return self.data_id, self.log.to_dict()

    @classmethod
    def from_legacy_format(cls, data_id, nested_dict, mask=None):
        """
        Create a BatchInfo instance from the legacy format.

        Args:
            data_id: Batch identifier
            nested_dict: Nested dictionary mapping module -> log_type -> tensor
            mask: Optional mask tensor

        Returns:
            BatchInfo: New BatchInfo instance with the data from the legacy format
        """
        batch_info = cls()
        batch_info.data_id = data_id
        batch_info.mask = mask
        batch_info.log = TensorLog.from_dict(nested_dict)
        return batch_info