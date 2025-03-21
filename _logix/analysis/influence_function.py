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
#
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
from einops import reduce
from tqdm import tqdm

from _logix.analysis.influence_function_utils import (
    cross_dot_product,
    merge_influence_results,
    precondition_kfac,
    precondition_raw,
)
from _logix.state import LogIXState
from _logix.tensor_log import TensorLog
from _logix.utils import get_logger, synchronize_tensor_log, flatten_tensor_log


class InfluenceFunction:
    """
    Computes influence functions to determine which training examples
    most affect model predictions on specific test examples.
    """
    def __init__(self, state: LogIXState):
        """
        Initialize the InfluenceFunction with a state.

        Args:
            state: LogIXState containing model statistics
        """
        # state
        self._state = state

        self.influence_scores = {}
        self.self_influence_scores = {}

    def get_influence_scores(self):
        """
        Get computed influence scores.

        Returns:
            Dict: Influence scores
        """
        return self.influence_scores

    def get_self_influence_scores(self):
        """
        Get computed self-influence scores.

        Returns:
            Dict: Self-influence scores
        """
        return self.self_influence_scores

    @torch.no_grad()
    def precondition(
        self,
        src_log: Tuple[List[str], Union[TensorLog, Dict, torch.Tensor]],
        damping: Optional[float] = None,
        hessian: Optional[str] = "auto",
    ) -> Tuple[List[str], Union[TensorLog, torch.Tensor]]:
        """
        Precondition gradients using the Hessian.

        Args:
            src_log: Tuple of (src_ids, log)
            damping: Damping parameter for preconditioning
            hessian: Type of Hessian approximation ("auto", "kfac", "raw")

        Returns:
            Tuple: (src_ids, preconditioned_gradients)
        """
        assert hessian in ["auto", "kfac", "raw"], f"Invalid hessian {hessian}"

        src_ids, src = src_log

        # Handle non-TensorLog inputs
        if not isinstance(src, TensorLog):
            if isinstance(src, dict):
                src = TensorLog.from_dict(src)
            elif isinstance(src, torch.Tensor):
                return src_log

        # Check if we have covariance statistics
        covariance_state = self._state.get_covariance_state()
        if not covariance_state:
            get_logger().warning(
                "Covariance state is empty. No preconditioning applied.\n"
            )
            return src_log

        # Check that we have covariance for all modules in src
        modules_missing_cov = []
        for module_name in src.get_all_modules():
            key = (module_name, "forward")
            if key not in covariance_state:
                modules_missing_cov.append(module_name)

        if modules_missing_cov:
            get_logger().warning(
                f"Covariance missing for modules: {modules_missing_cov}. No preconditioning applied.\n"
            )
            return src_log

        # Choose the appropriate preconditioning function
        precondition_fn = precondition_kfac
        if hessian == "raw" or (
            hessian == "auto" and all((module_name, "grad") in covariance_state
                                   for module_name in src.get_all_modules())
        ):
            precondition_fn = precondition_raw

        # Apply preconditioning
        preconditioned_grad = precondition_fn(
            src=(src_ids, src), state=self._state, damping=damping
        )

        return preconditioned_grad

    @torch.no_grad()
    def compute_influence(
        self,
        src_log: Tuple[List[str], Union[TensorLog, Dict, torch.Tensor]],
        tgt_log: Tuple[List[str], Union[TensorLog, Dict, torch.Tensor]],
        mode: Optional[str] = "dot",
        precondition: Optional[bool] = True,
        hessian: Optional[str] = "auto",
        influence_groups: Optional[List[str]] = None,
        damping: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute influence scores between two gradient logs.

        Args:
            src_log: Tuple of (src_ids, source_gradients)
            tgt_log: Tuple of (tgt_ids, target_gradients)
            mode: Influence function mode ("dot", "l2", "cosine")
            precondition: Whether to precondition the gradients
            hessian: Type of Hessian approximation
            influence_groups: Optional list of module groups to track separately
            damping: Damping parameter for preconditioning

        Returns:
            Dict: Influence results
        """
        assert mode in ["dot", "l2", "cosine"], f"Invalid mode: {mode}"

        result = {}

        # Preprocess logs to ensure they're in TensorLog format
        src_ids, src = src_log
        tgt_ids, tgt = tgt_log

        if not isinstance(src, TensorLog) and isinstance(src, dict):
            src = TensorLog.from_dict(src)

        if not isinstance(tgt, TensorLog) and isinstance(tgt, dict):
            tgt = TensorLog.from_dict(tgt)

        # Apply preconditioning if requested
        if precondition:
            _, src = self.precondition(
                src_log=(src_ids, src), damping=damping, hessian=hessian
            )

        # Initialize influence scores
        total_influence = {"total": 0}
        for influence_group in influence_groups or []:
            total_influence[influence_group] = 0

        # Handle special case where target is a tensor
        if not isinstance(tgt, TensorLog) and isinstance(tgt, torch.Tensor):
            model_path = self._state.get_state("model_module")["path"]
            if isinstance(src, TensorLog):
                src_flat = flatten_tensor_log(src, model_path)
            else:
                src_flat = src

            tgt = tgt.to(device=src_flat.device)
            total_influence["total"] += cross_dot_product(src_flat, tgt)
        else:
            # Ensure src and tgt are on the same device
            synchronize_tensor_log(tgt, src.get_all_modules()[0])

            # Compute influence for each module
            for module_name in src.get_all_modules():
                src_grad = src.get(module_name, "grad")
                tgt_grad = tgt.get(module_name, "grad")

                if src_grad is None or tgt_grad is None:
                    continue

                module_influence = cross_dot_product(src_grad, tgt_grad)
                total_influence["total"] += module_influence

                # Group-specific influence
                if influence_groups is not None:
                    groups = [g for g in influence_groups if g in module_name]
                    for group in groups:
                        total_influence[group] += module_influence

        # Normalize influence scores based on mode
        if mode == "cosine":
            tgt_norm = self.compute_self_influence(
                tgt_log,
                precondition=True,
                hessian=hessian,
                influence_groups=influence_groups,
                damping=damping,
            ).pop("influence")

            for key in total_influence.keys():
                tgt_norm_key = tgt_norm if influence_groups is None else tgt_norm[key]
                total_influence[key] /= torch.sqrt(tgt_norm_key.unsqueeze(0))
        elif mode == "l2":
            tgt_norm = self.compute_self_influence(
                tgt_log,
                precondition=True,
                hessian=hessian,
                influence_groups=influence_groups,
                damping=damping,
            ).pop("influence")

            for key in total_influence.keys():
                tgt_norm_key = tgt_norm if influence_groups is None else tgt_norm[key]
                total_influence[key] -= 0.5 * tgt_norm_key.unsqueeze(0)

        # Move influence scores to CPU to save memory
        for key, value in total_influence.items():
            assert value.shape[0] == len(src_ids)
            assert value.shape[1] == len(tgt_ids)
            total_influence[key] = value.cpu()

        result["src_ids"] = list(src_ids)
        result["tgt_ids"] = list(tgt_ids)
        result["influence"] = (
            total_influence.pop("total")
            if influence_groups is None
            else total_influence
        )

        return result

    @torch.no_grad()
    def compute_self_influence(
        self,
        src_log: Tuple[List[str], Union[TensorLog, Dict, torch.Tensor]],
        precondition: Optional[bool] = True,
        hessian: Optional[str] = "auto",
        influence_groups: Optional[List[str]] = None,
        damping: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute self-influence scores. This can be used for uncertainty estimation.

        Args:
            src_log: Tuple of (src_ids, source_gradients)
            precondition: Whether to precondition the gradients
            hessian: Type of Hessian approximation
            influence_groups: Optional list of module groups to track separately
            damping: Damping parameter for preconditioning

        Returns:
            Dict: Self-influence results
        """
        result = {}

        src_ids, src = src_log

        # Preprocess logs to ensure they're in TensorLog format
        if not isinstance(src, TensorLog) and isinstance(src, dict):
            src = TensorLog.from_dict(src)
        elif not isinstance(src, TensorLog) and isinstance(src, torch.Tensor):
            model_path = self._state.get_state("model_module")["path"]
            temp_log = TensorLog()
            for i, (module_name, log_type) in enumerate(model_path):
                temp_log.add(module_name, log_type, src)
            src = temp_log

        # Apply preconditioning if requested
        tgt = src
        if precondition:
            _, tgt = self.precondition(
                src_log=(src_ids, src), hessian=hessian, damping=damping
            )

        # Initialize influence scores
        total_influence = {"total": 0}
        for influence_group in influence_groups or []:
            total_influence[influence_group] = 0

        # Compute self-influence scores for each module
        for module_name in src.get_all_modules():
            src_grad = src.get(module_name, "grad")
            tgt_grad = tgt.get(module_name, "grad") if tgt is not src else src_grad

            if src_grad is None or tgt_grad is None:
                continue

            # Element-wise multiplication and sum for self-influence
            module_influence = reduce(
                src_grad * tgt_grad, "n a b -> n", "sum"
            ).reshape(-1)

            total_influence["total"] += module_influence

            # Group-specific influence
            if influence_groups is not None:
                groups = [g for g in influence_groups if g in module_name]
                for group in groups:
                    total_influence[group] += module_influence

        # Move influence scores to CPU to save memory
        for key, value in total_influence.items():
            assert len(value) == len(src_ids)
            total_influence[key] = value.cpu()

        result["src_ids"] = src_ids
        result["influence"] = (
            total_influence.pop("total")
            if influence_groups is None
            else total_influence
        )

        return result

    def compute_influence_all(
        self,
        src_log: Optional[Tuple[List[str], Union[TensorLog, Dict, torch.Tensor]]] = None,
        loader: Optional[torch.utils.data.DataLoader] = None,
        mode: Optional[str] = "dot",
        precondition: Optional[bool] = True,
        save: Optional[bool] = False,
        hessian: Optional[str] = "auto",
        influence_groups: Optional[List[str]] = None,
        damping: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute influence scores against all training data in the log.

        Args:
            src_log: Tuple of (src_ids, source_gradients)
            loader: DataLoader of training data
            mode: Influence function mode
            precondition: Whether to precondition the gradients
            save: Whether to save results to the influence_scores attribute
            hessian: Type of Hessian approximation
            influence_groups: Optional list of module groups to track separately
            damping: Damping parameter for preconditioning

        Returns:
            Dict: Influence results for all training data
        """
        if precondition:
            src_log = self.precondition(src_log, hessian=hessian, damping=damping)

        result_all = {}
        for tgt_log in tqdm(loader, desc="Compute IF"):
            result = self.compute_influence(
                src_log=src_log,
                tgt_log=tgt_log,
                mode=mode,
                precondition=False,  # Already preconditioned above
                hessian=hessian,
                influence_groups=influence_groups,
                damping=damping,
            )
            merge_influence_results(result_all, result, axis="tgt")

        if save:
            merge_influence_results(self.influence_scores, result_all, axis="src")

        return result_all