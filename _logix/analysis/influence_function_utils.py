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

from typing import Dict, List, Optional, Tuple, Any

import torch
from einops import einsum, rearrange, reduce

from _logix.state import LogIXState
from _logix.statistic.utils import make_2d
from _logix.tensor_log import TensorLog


def precondition_kfac(
    src: Tuple[List[str], TensorLog],
    state: LogIXState,
    damping: Optional[float] = None,
) -> Tuple[List[str], TensorLog]:
    """
    Precondition gradients using KFAC approximation of the Hessian.

    Args:
        src: Tuple of (src_ids, TensorLog)
        state: LogIXState containing eigenvalue and eigenvector information
        damping: Damping parameter for numerical stability

    Returns:
        Tuple: (src_ids, preconditioned_gradients)
    """
    src_ids, src_log = src
    preconditioned = TensorLog()
    cov_eigval, cov_eigvec = state.get_covariance_svd_state()

    for module_name in src_log.get_all_modules():
        src_grad = src_log.get(module_name, "grad")
        if src_grad is None:
            continue

        device = src_grad.device

        key_fwd = (module_name, "forward")
        key_bwd = (module_name, "backward")

        # Skip modules without eigendecomposition
        if key_fwd not in cov_eigvec or key_bwd not in cov_eigvec:
            continue

        fwd_eigvec = cov_eigvec[key_fwd].to(device=device)
        bwd_eigvec = cov_eigvec[key_bwd].to(device=device)

        # Reconstruct the full eigenvalue matrix with the damping factor added
        if isinstance(cov_eigval, Dict):
            if key_fwd in cov_eigval and key_bwd in cov_eigval:
                fwd_eigval = cov_eigval[key_fwd]
                bwd_eigval = cov_eigval[key_bwd]
                full_eigval = torch.outer(bwd_eigval, fwd_eigval).to(device=device)
            else:
                continue
        else:
            full_eigval = cov_eigval[(module_name, "ekfac")].to(device=device)

        if damping is None:
            damping = 0.1 * torch.mean(full_eigval)
        full_eigval += damping

        # Precondition the gradient using eigenvectors and eigenvalues
        rotated_grad = einsum(
            bwd_eigvec.t(),
            src_grad,
            fwd_eigvec,
            "a b, batch b c, c d -> batch a d",
        )
        prec_rotated_grad = rotated_grad / full_eigval
        preconditioned_grad = einsum(
            bwd_eigvec,
            prec_rotated_grad,
            fwd_eigvec.t(),
            "a b, batch b c, c d -> batch a d",
        )

        # Add to preconditioned TensorLog
        preconditioned.add(module_name, "grad", preconditioned_grad)

    return (src_ids, preconditioned)


def precondition_raw(
    src: Tuple[List[str], TensorLog],
    state: LogIXState,
    damping: Optional[float] = None,
) -> Tuple[List[str], TensorLog]:
    """
    Precondition gradients using raw covariance inverse.

    Args:
        src: Tuple of (src_ids, TensorLog)
        state: LogIXState containing covariance inverse information
        damping: Damping parameter for numerical stability

    Returns:
        Tuple: (src_ids, preconditioned_gradients)
    """
    src_ids, src_log = src
    preconditioned = TensorLog()
    cov_inverse = state.get_covariance_inverse_state(damping=damping)

    for module_name in src_log.get_all_modules():
        src_grad = src_log.get(module_name, "grad")
        if src_grad is None:
            continue

        device = src_grad.device
        key = (module_name, "grad")

        # Skip modules without covariance inverse
        if key not in cov_inverse:
            continue

        grad_cov_inverse = cov_inverse[key].to(device=device)
        original_shape = src_grad.shape

        # Reshape, apply preconditioner, and reshape back
        preconditioned_grad = (
            make_2d(src_grad, None, "grad") @ grad_cov_inverse
        ).reshape(original_shape)

        # Add to preconditioned TensorLog
        preconditioned.add(module_name, "grad", preconditioned_grad)

    return (src_ids, preconditioned)


def cross_dot_product(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross dot product between two tensors.

    Args:
        src: Source tensor of shape (n, ...)
        tgt: Target tensor of shape (m, ...)

    Returns:
        torch.Tensor: Cross dot product of shape (n, m)
    """
    assert src.shape[1:] == tgt.shape[1:]
    src_expanded = rearrange(src, "n ... -> n 1 ...")
    tgt_expanded = rearrange(tgt, "m ... -> 1 m ...")
    dot_product_result = reduce(
        src_expanded * tgt_expanded,
        "n m ... -> n m",
        "sum",
    )

    return dot_product_result


def merge_influence_results(
    result_all: Dict[str, Any],
    result: Dict[str, Any],
    axis: str = "tgt",
) -> None:
    """
    Merge influence results along the specified axis.

    Args:
        result_all: Accumulated results dictionary to update
        result: New results dictionary to merge
        axis: Axis along which to merge ("src" or "tgt")
    """
    assert axis in ["src", "tgt"], f"Unsupported axis {axis}."

    # If merged result is empty, just copy the result and return
    if not result_all:
        result_all.update(result)
        return

    dim = int(axis == "tgt")
    id_key = f"{axis}_ids"

    result_all[id_key].extend(result[id_key])
    if isinstance(result["influence"], dict):
        for key in result_all["influence"].keys():
            result_all["influence"][key] = torch.cat(
                [result_all["influence"][key], result["influence"][key]], dim=dim
            )
    else:
        assert isinstance(result["influence"], torch.Tensor)
        result_all["influence"] = torch.cat(
            [result_all["influence"], result["influence"]], dim=dim
        )