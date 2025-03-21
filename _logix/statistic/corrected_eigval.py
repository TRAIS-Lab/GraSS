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


class CorrectedEigval:
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
        Update the corrected eigenvalue state.

        Args:
            state: LogIXState object
            binfo: BatchInfo object
            module: The module being logged
            module_name: The name of the module
            log_type: The type of log (forward, backward, grad)
            data: Optional tensor data
            cpu_offload: Whether to offload tensors to CPU
        """
        # Initialize required states if not already present
        if not hasattr(state, "covariance_eigvec_state"):
            state.covariance_svd()
            state.register_state("ekfac_eigval_state", synchronize=True, save=True)
            state.register_state("ekfac_counter", synchronize=True, save=False)
            state.register_normalize_pair("ekfac_eigval_state", "ekfac_counter")

        covariance_eigvec_state = state.covariance_eigvec_state
        covariance_eigval_state = state.covariance_eigval_state
        ekfac_eigval_state = state.ekfac_eigval_state
        ekfac_counter = state.ekfac_counter

        # Get data from BatchInfo if not provided
        if data is None:
            data = binfo.log.get(module_name, "grad")
            if data is None:
                return

        key_fwd = (module_name, "forward")
        key_bwd = (module_name, "backward")
        key_ekfac = (module_name, "ekfac")

        # Initialize ekfac_eigval_state if necessary
        if key_ekfac not in ekfac_eigval_state:
            device = data.device if not cpu_offload else "cpu"
            dtype = data.dtype
            ekfac_eigval_state[key_ekfac] = torch.zeros(
                data.shape[-2], data.shape[-1], device=device, dtype=dtype
            )
            ekfac_counter[key_ekfac] = 0

        data = data.detach()

        if cpu_offload:
            # Move to GPU for efficient computation, then back to CPU
            eigvec_fwd_gpu = covariance_eigvec_state[key_fwd].to(device=data.device)
            eigvec_bwd_gpu = covariance_eigvec_state[key_bwd].to(device=data.device)
            ekfac_eigval_state_gpu = ekfac_eigval_state[key_ekfac].to(device=data.device)

            # Compute rotated gradients
            rotated_grads = torch.matmul(data, eigvec_fwd_gpu)

            # Update eigenvalues
            for rotated_grad in rotated_grads:
                weight = torch.matmul(eigvec_bwd_gpu.t(), rotated_grad)
                ekfac_eigval_state_gpu.add_(weight.square_())

            # Move back to CPU
            ekfac_eigval_state[key_ekfac] = ekfac_eigval_state_gpu.to(
                device="cpu", non_blocking=True
            )

            # Free GPU memory
            del eigvec_fwd_gpu
            del eigvec_bwd_gpu
            del ekfac_eigval_state_gpu
        else:
            # Compute on current device
            rotated_grads = torch.matmul(data, covariance_eigvec_state[key_fwd])

            for rotated_grad in rotated_grads:
                weight = torch.matmul(covariance_eigvec_state[key_bwd].t(), rotated_grad)
                ekfac_eigval_state[key_ekfac].add_(weight.square_())

        # Update counter
        ekfac_counter[key_ekfac] += len(data)