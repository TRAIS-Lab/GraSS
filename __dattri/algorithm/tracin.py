"""This module implements the TracIn attributor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import List, Optional, Union

    from dattri.task import AttributionTask

import torch
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm

from _dattri.utlis import compute_pairwise_distance_metrics

from _dattri.func.projection import random_project

from dattri.algorithm.base import BaseAttributor
from dattri.algorithm.utils import _check_shuffle

DEFAULT_PROJECTOR_KWARGS = {
    "proj_dim": 512,
    "proj_max_batch_size": 32,
    "proj_seed": 0,
    "device": "cpu",
    "use_half_precision": False,
}

class TracInAttributor(BaseAttributor):
    """TracIn attributor."""

    def __init__(
        self,
        task: AttributionTask,
        weight_list: Tensor,
        normalized_grad: bool,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the TracIn attributor.

        Args:
            task (AttributionTask): The task to be attributed. Please refer to the
                `AttributionTask` for more details.
            weight_list (Tensor): The weight used for the "weighted sum". For
                TracIn/CosIn, this will contain a list of learning rates at each ckpt;
                for Grad-Dot/Grad-Cos, this will be a list of ones.
            normalized_grad (bool): Whether to apply normalization to gradients.
            projector_kwargs (Optional[Dict[str, Any]]): The keyword arguments for the
                projector.
            layer_name (Optional[Union[str, List[str]]]): The name of the layer to be
                used to calculate the train/test representations. If None, full
                parameters are used. This should be a string or a list of strings
                if multiple layers are needed. The name of layer should follow the
                key of model.named_parameters(). Default: None.
            device (str): The device to run the attributor. Default is cpu.
        """
        self.task = task
        self.weight_list = weight_list
        self.projector_kwargs = DEFAULT_PROJECTOR_KWARGS
        if projector_kwargs is not None:
            self.projector_kwargs.update(projector_kwargs)
        self.normalized_grad = normalized_grad
        self.layer_name = layer_name
        self.device = device
        self.full_train_dataloader = None
        # to get per-sample gradients for a mini-batch of train/test samples
        self.grad_target_func = self.task.get_grad_target_func(in_dims=(None, 0))
        self.grad_loss_func = self.task.get_grad_loss_func(in_dims=(None, 0))

    def cache(self) -> None:
        """Precompute and cache some values for efficiency."""

    def attribute(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        sparsify: dict = None,
        sparse_check: bool = False,
        verbose=True,
    ) -> Tensor:
        """Calculate the influence of the training set on the test set.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The dataloader for
                training samples to calculate the influence. It can be a subset
                of the full training set if `cache` is called before. A subset
                means that only a part of the training set's influence is calculated.
                The dataloader should not be shuffled.
            test_dataloader (torch.utils.data.DataLoader): The dataloader for
                test samples to calculate the influence. The dataloader should not
                be shuffled.

        Raises:
            ValueError: The length of params_list and weight_list don't match.

        Returns:
            Tensor: The influence of the training set on the test set, with
                the shape of (num_train_samples, num_test_samples).
        """
        _check_shuffle(test_dataloader)
        _check_shuffle(train_dataloader)

        # check the length match between checkpoint list and weight list
        if len(self.task.get_checkpoints()) != len(self.weight_list):
            msg = "the length of checkpoints and weights lists don't match."
            raise ValueError(msg)

        # placeholder for the TDA result
        # should work for torch dataset without sampler
        tda_output = torch.zeros(
            size=(len(train_dataloader.sampler), len(test_dataloader.sampler)),
        )

        # Initialize variables to accumulate sparsity and distance metrics
        total_original_sparsity = 0.0
        total_projected_sparsity = 0.0
        distance_RE = []

        total_samples = 0

        # Check if sparsify is provided
        sparsify_method = sparsify.get('method', None) if sparsify else None

        # iterate over each checkpoint (each ensemble)
        for ckpt_idx, ckpt_weight in zip(
            range(len(self.task.get_checkpoints())),
            self.weight_list,
        ):
            parameters, _ = self.task.get_param(
                index=ckpt_idx,
                layer_name=self.layer_name,
            )
            if self.layer_name is not None:
                self.grad_target_func = self.task.get_grad_target_func(
                    in_dims=(None, 0),
                    layer_name=self.layer_name,
                    # ckpt_idx=ckpt_idx,
                )
                self.grad_loss_func = self.task.get_grad_loss_func(
                    in_dims=(None, 0),
                    layer_name=self.layer_name,
                    # ckpt_idx=ckpt_idx,
                )

            for train_batch_idx, train_batch_data_ in enumerate(
                tqdm(
                    train_dataloader,
                    desc="calculating gradient of training set...",
                    leave=False,
                    disable=not verbose,
                ),
            ):
                # move to device
                train_batch_data = tuple(
                    data.to(self.device) for data in train_batch_data_
                )
                # get gradient of train
                grad_t = self.grad_loss_func(parameters, train_batch_data)
                grad_t = torch.nan_to_num(grad_t)

                # Apply sparsification based on method specified
                if sparsify_method == 'threshold':
                    eps = sparsify.get('param', float('inf')) if sparsify else float('inf')
                    grad_t[torch.abs(grad_t) < eps] = 0
                elif sparsify_method == 'random':
                    drop_rate = sparsify.get('param', 0.0) if sparsify else 0.0
                    batch_size, num_params = grad_t.size(0), grad_t.size(1)
                    num_elements_to_drop = int(drop_rate * num_params)

                    if num_elements_to_drop > 0:
                        for i in range(batch_size):
                            indices_to_drop = torch.randperm(num_params)[:num_elements_to_drop]
                            grad_t[i, indices_to_drop] = 0

                # save grad_t to disk for debugging
                torch.save(grad_t, f"grad_t_{ckpt_idx}_{train_batch_idx}.pt")

                # Projected gradient
                grad_p = (
                    random_project(
                        grad_t,
                        train_batch_data[0].shape[0],
                        **self.projector_kwargs,
                    )(grad_t, ensemble_id=ckpt_idx)
                    .clone()
                    .detach()
                )

                # normalize the projected gradient according to the JL lemma (1 / sqrt(proj_dim))
                grad_p /= self.projector_kwargs["proj_dim"] ** 0.5

                if sparse_check:
                    # Calculate and accumulate sparsity for the batch
                    batch_size = grad_t.size(0)
                    original_sparsity = (grad_t == 0).float().mean().item()
                    projected_sparsity = (grad_p == 0).float().mean().item()
                    total_original_sparsity += original_sparsity * batch_size
                    total_projected_sparsity += projected_sparsity * batch_size

                    # Calculate pairwise distances for this batch
                    # relative_error, rmse, stress = compute_pairwise_distance_metrics(grad_t, grad_p)
                    relative_error = compute_pairwise_distance_metrics(grad_t, grad_p)
                    distance_RE.append(relative_error)
                    # distance_rmse.append(rmse)
                    # distance_stress.append(stress)

                    total_samples += batch_size

                # if self.projector_kwargs is not None:
                #     # define the projector for this batch of data
                #     self.train_random_project = random_project(
                #         grad_t,
                #         # get the batch size, prevent edge case
                #         train_batch_data[0].shape[0],
                #         **self.projector_kwargs,
                #     )
                #     # param index as ensemble id
                #     train_batch_grad = self.train_random_project(
                #         torch.nan_to_num(grad_t),
                #         ensemble_id=ckpt_idx,
                #     )
                # else:
                #     train_batch_grad = torch.nan_to_num(grad_t)

                train_batch_grad = grad_p

                for test_batch_idx, test_batch_data_ in enumerate(
                    tqdm(
                        test_dataloader,
                        desc="calculating gradient of test set...",
                        leave=False,
                        disable=not verbose,
                    ),
                ):
                    # move to device
                    test_batch_data = tuple(
                        data.to(self.device) for data in test_batch_data_
                    )
                    # get gradient of test
                    grad_t = self.grad_target_func(parameters, test_batch_data)
                    grad_t = torch.nan_to_num(grad_t)

                    # Apply sparsification based on method specified
                    if sparsify_method == 'threshold':
                        eps = sparsify.get('param', float('inf')) if sparsify else float('inf')
                        grad_t[torch.abs(grad_t) < eps] = 0
                    elif sparsify_method == 'random':
                        drop_rate = sparsify.get('param', 0.0) if sparsify else 0.0
                        batch_size, num_params = grad_t.size(0), grad_t.size(1)
                        num_elements_to_drop = int(drop_rate * num_params)

                        if num_elements_to_drop > 0:
                            for i in range(batch_size):
                                indices_to_drop = torch.randperm(num_params)[:num_elements_to_drop]
                                grad_t[i, indices_to_drop] = 0

                    grad_p = (
                        random_project(
                            grad_t,
                            test_batch_data[0].shape[0],
                            **self.projector_kwargs,
                        )(grad_t, ensemble_id=ckpt_idx)
                        .clone()
                        .detach()
                    )

                    # normalize the projected gradient according to the JL lemma (1 / sqrt(proj_dim))
                    grad_p /= self.projector_kwargs["proj_dim"] ** 0.5

                    # if self.projector_kwargs is not None:
                    #     # define the projector for this batch of data
                    #     self.test_random_project = random_project(
                    #         grad_t,
                    #         test_batch_data[0].shape[0],
                    #         **self.projector_kwargs,
                    #     )

                    #     test_batch_grad = self.test_random_project(
                    #         torch.nan_to_num(grad_t),
                    #         ensemble_id=ckpt_idx,
                    #     )
                    # else:
                    #     test_batch_grad = torch.nan_to_num(grad_t)

                    test_batch_grad = grad_p

                    # results position based on batch info
                    row_st = train_batch_idx * train_dataloader.batch_size
                    row_ed = min(
                        (train_batch_idx + 1) * train_dataloader.batch_size,
                        len(train_dataloader.sampler),
                    )

                    col_st = test_batch_idx * test_dataloader.batch_size
                    col_ed = min(
                        (test_batch_idx + 1) * test_dataloader.batch_size,
                        len(test_dataloader.sampler),
                    )
                    # accumulate the TDA score in corresponding positions (blocks)
                    if self.normalized_grad:
                        tda_output[row_st:row_ed, col_st:col_ed] += (
                            (
                                normalize(train_batch_grad)
                                @ normalize(test_batch_grad).T
                                * ckpt_weight
                            )
                            .detach()
                            .cpu()
                        )
                    else:
                        tda_output[row_st:row_ed, col_st:col_ed] += (
                            (train_batch_grad @ test_batch_grad.T * ckpt_weight)
                            .detach()
                            .cpu()
                        )

        if sparse_check and total_samples > 0:
            # Compute final average sparsity
            avg_original_sparsity = total_original_sparsity / total_samples
            avg_projected_sparsity = total_projected_sparsity / total_samples
            avg_distance_RE = sum(distance_RE) / len(distance_RE) if distance_RE else 0
            # avg_distance_rmse = sum(distance_rmse) / len(distance_rmse) if distance_rmse else 0
            # avg_distance_stress = sum(distance_stress) / len(distance_stress) if distance_stress else 0

            print(f"Average Sparsity of Original Gradients: {avg_original_sparsity:.4f}")
            print(f"Average Sparsity of Projected Gradients: {avg_projected_sparsity:.4f}")
            print(f"Average Distance Relative Error (Original vs Projected): {avg_distance_RE:.4f}")
            # print(f"Average Distance RMSE (Original vs Projected): {avg_distance_rmse:.4f}")
            # print(f"Average Distance Stress (Original vs Projected): {avg_distance_stress:.4f}")

        return tda_output