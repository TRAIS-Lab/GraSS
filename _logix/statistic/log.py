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

from typing import Optional

import torch
import torch.nn as nn

from _logix.batch_info import BatchInfo
from _logix.state import LogIXState


class Log:
    @staticmethod
    @torch.no_grad()
    def update(
        state: LogIXState,
        binfo: BatchInfo,
        module: nn.Module,
        module_name: str,
        log_type: str,
        data: Optional[torch.Tensor] = None,
        cpu_offload: Optional[bool] = False,
    ):
        """
        Add log tensor to BatchInfo's tensor log.

        Args:
            state: LogIXState object
            binfo: BatchInfo object to update
            module: The module being logged
            module_name: The name of the module
            log_type: The type of log (forward, backward, grad)
            data: Optional tensor data to log
            cpu_offload: Whether to offload tensors to CPU
        """
        # If data is provided, add it directly to the tensor log
        if data is not None:
            # Move to CPU if requested
            if cpu_offload:
                data = data.detach().cpu()

            # Add data to the tensor log
            binfo.log.add(module_name, log_type, data)