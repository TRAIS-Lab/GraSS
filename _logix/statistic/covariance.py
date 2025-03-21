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
from _logix.statistic.utils import make_2d


class Covariance:
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
        Update the covariance state.

        Args:
            state: LogIXState object
            binfo: BatchInfo object
            module: The module being logged
            module_name: The name of the module
            log_type: The type of log (forward, backward, grad)
            data: Optional tensor data
            cpu_offload: Whether to offload tensors to CPU
        """
        covariance_state = state.covariance_state
        covariance_counter = state.covariance_counter

        # Get data from BatchInfo if not provided
        if data is None:
            data = binfo.log.get(module_name, log_type)
            if data is None:
                return

        # Extract and reshape data to 2d tensor for covariance computation
        batch_size = data.size(0)
        data = make_2d(data, module, log_type).detach()

        # Create key for covariance_state dictionary
        key = (module_name, log_type)

        # Initialize covariance state if necessary
        if key not in covariance_state:
            device = data.device if not cpu_offload else "cpu"
            dtype = data.dtype
            covariance_state[key] = torch.zeros(
                data.shape[-1], data.shape[-1], device=device, dtype=dtype
            )
            covariance_counter[key] = 0

        # Update covariance state
        if cpu_offload:
            # Move to GPU for efficient computation, then back to CPU
            covariance_state_gpu = covariance_state[key].to(device=data.device)
            covariance_state_gpu.addmm_(data.t(), data)
            covariance_state[key] = covariance_state_gpu.to(device="cpu", non_blocking=True)
        else:
            covariance_state[key].addmm_(data.t(), data)

        # Update covariance counter
        if binfo.mask is None or log_type == "grad":
            covariance_counter[key] += batch_size
        else:
            covariance_counter[key] += binfo.mask.sum().item()