"""This module implements the TracIn attributor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import List, Optional, Union

    from ..task import AttributionTask

import torch
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm

from ..func.projection import random_project

from .base import BaseAttributor
from .utils import _check_shuffle

DEFAULT_PROJECTOR_KWARGS = None

class TracInAttributor(BaseAttributor):
    """TracIn attributor."""

    def __init__(
        self,
        task: AttributionTask,
        weight_list: Tensor,
        normalized_grad: bool,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        mode: str = "default",
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
        self.projector_kwargs = projector_kwargs
        self.normalized_grad = normalized_grad
        self.layer_name = layer_name
        self.device = device
        self.mode = mode
        self.full_train_dataloader = None
        # to get per-sample gradients for a mini-batch of train/test samples
        self.grad_target_func = self.task.get_grad_target_func(in_dims=(None, 0))
        self.grad_loss_func = self.task.get_grad_loss_func(in_dims=(None, 0))

    def cache(
        self,
        full_train_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """Cache the training gradients for more efficient attribution.

        Args:
            full_train_dataloader (torch.utils.data.DataLoader): The dataloader
                with full training samples for gradient calculation.
        """
        _check_shuffle(full_train_dataloader)
        self.full_train_dataloader = full_train_dataloader

        # Store cached gradients for each checkpoint
        self.cached_train_grads = []

        for ckpt_idx in range(len(self.task.get_checkpoints())):
            parameters, _ = self.task.get_param(
                ckpt_idx=ckpt_idx,
                layer_name=self.layer_name,
            )

            if self.layer_name is not None:
                self.grad_loss_func = self.task.get_grad_loss_func(
                    in_dims=(None, 0),
                    layer_name=self.layer_name,
                    ckpt_idx=ckpt_idx,
                )

            train_batch_grads = None
            for train_batch_idx, train_data in enumerate(
                tqdm(
                    full_train_dataloader,
                    desc=f"caching gradients for checkpoint {ckpt_idx}...",
                    leave=False,
                ),
            ):
                # Handle different data formats
                if isinstance(train_data, (tuple, list)):
                    train_batch_data = tuple(
                        data.to(self.device) for data in train_data
                    )
                else:
                    train_batch_data = train_data

                # Compute gradients
                grad_t = self.grad_loss_func(parameters, train_batch_data)
                grad_t = torch.nan_to_num(grad_t)

                # Apply projection if specified
                if self.projector_kwargs is not None:
                    train_random_project = random_project(
                        grad_t,
                        grad_t.shape[0],
                        **self.projector_kwargs,
                    )
                    grad_t = train_random_project(
                        grad_t,
                        ensemble_id=ckpt_idx,
                    )

                # Apply normalization if specified
                if self.normalized_grad:
                    grad_t = normalize(grad_t)

                if train_batch_grads is None:
                    total_samples = len(full_train_dataloader.sampler)
                    train_batch_grads = torch.zeros((total_samples, *grad_t.shape[1:]), device=self.device)

                col_st = train_batch_idx * full_train_dataloader.batch_size
                col_ed = min(
                    (train_batch_idx + 1) * full_train_dataloader.batch_size,
                    len(full_train_dataloader.sampler),
                )
                train_batch_grads[col_st:col_ed] = grad_t.detach()

            self.cached_train_grads.append(train_batch_grads)

    def attribute(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Tensor:
        """Calculate the influence of the training set on the test set.

        Args:
            test_dataloader (torch.utils.data.DataLoader): The dataloader for
                test samples to calculate the influence. The dataloader should not
                be shuffled.
            train_dataloader (Optional[torch.utils.data.DataLoader]): The dataloader for
                training samples to calculate the influence. If None and cache was called,
                uses cached gradients. If provided with cached gradients, raises an error.
                The dataloader should not be shuffled.

        Raises:
            ValueError: If the length of params_list and weight_list don't match,
                or if train_dataloader is provided when cached gradients exist.

        Returns:
            Tensor: The influence of the training set on the test set, with
                the shape of (num_train_samples, num_test_samples).
        """
        _check_shuffle(test_dataloader)
        if train_dataloader is not None:
            _check_shuffle(train_dataloader)

        # Check if trying to use both cache and new training data
        if train_dataloader is not None and self.full_train_dataloader is not None:
            message = "You have cached a training loader by .cache()\
                       and you are trying to attribute a different training loader.\
                       If this new training loader is a subset of the cached training\
                       loader, please don't input the training dataloader in\
                       .attribute() and directly use index to select the corresponding\
                       scores."
            raise ValueError(message)
        if train_dataloader is None and self.full_train_dataloader is None:
            message = "You did not state a training loader in .attribute() and you\
                       did not cache a training loader by .cache(). Please provide a\
                       training loader or cache a training loader."
            raise ValueError(message)

        # Check checkpoint and weight list lengths match
        if len(self.task.get_checkpoints()) != len(self.weight_list):
            raise ValueError("The length of checkpoints and weights lists don't match.")

        if self.mode == "default": # process batch-wise
            tda_output = self.attribute_default(test_dataloader, train_dataloader)
        elif self.mode == "iterate":
            tda_output = self.attribute_iterate(test_dataloader, train_dataloader)

        return tda_output

    def attribute_default(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Tensor:
        # Initialize output tensor
        if train_dataloader is not None:
            num_train = len(train_dataloader.sampler)
        else:
            num_train = len(self.full_train_dataloader.sampler)

        tda_output = torch.zeros(
            size=(num_train, len(test_dataloader.sampler)),
        )

        for ckpt_idx, ckpt_weight in enumerate(self.weight_list):
            parameters, _ = self.task.get_param(
                ckpt_idx=ckpt_idx,
                layer_name=self.layer_name,
            )

            if self.layer_name is not None:
                self.grad_target_func = self.task.get_grad_target_func(
                    in_dims=(None, 0),
                    layer_name=self.layer_name,
                    ckpt_idx=ckpt_idx,
                )
                if train_dataloader is not None:
                    self.grad_loss_func = self.task.get_grad_loss_func(
                        in_dims=(None, 0),
                        layer_name=self.layer_name,
                        ckpt_idx=ckpt_idx,
                    )

            if train_dataloader is not None:
                train_grads = None
                for train_batch_idx, train_batch_data_ in enumerate(
                    tqdm(
                        train_dataloader,
                        desc="calculating gradient of training set...",
                        leave=False,
                    ),
                ):
                    # Process training batch
                    if isinstance(train_batch_data_, (tuple, list)):
                        train_batch_data = tuple(
                            data.to(self.device) for data in train_batch_data_
                        )
                    else:
                        train_batch_data = train_batch_data_

                    # Compute gradients
                    grad_t = self.grad_loss_func(parameters, train_batch_data)
                    grad_t = torch.nan_to_num(grad_t)

                    # Apply projection if specified
                    if self.projector_kwargs is not None:
                        train_random_project = random_project(
                            grad_t,
                            grad_t.shape[0],
                            **self.projector_kwargs,
                        )
                        grad_t = train_random_project(
                            grad_t,
                            ensemble_id=ckpt_idx,
                        )

                    # Apply normalization if specified
                    if self.normalized_grad:
                        grad_t = normalize(grad_t)

                    if train_grads is None:
                        total_samples = len(train_dataloader.sampler)
                        train_grads = torch.zeros((total_samples, *grad_t.shape[1:]), device=self.device)

                    col_st = train_batch_idx * train_dataloader.batch_size
                    col_ed = min(
                        (train_batch_idx + 1) * train_dataloader.batch_size,
                        len(train_dataloader.sampler),
                    )
                    train_grads[col_st:col_ed] = grad_t.detach()
            else:
                # Use cached gradients
                train_grads = self.cached_train_grads[ckpt_idx].to(self.device)

            # Process test data batches
            for test_batch_idx, test_batch_data_ in enumerate(
                tqdm(
                    test_dataloader,
                    desc="calculating gradient of test set...",
                    leave=False,
                ),
            ):
                # Process test batch
                if isinstance(test_batch_data_, (tuple, list)):
                    test_batch_data = tuple(
                        data.to(self.device) for data in test_batch_data_
                    )
                else:
                    test_batch_data = test_batch_data_

                # Compute test gradients
                grad_t = self.grad_target_func(parameters, test_batch_data)
                torch.nan_to_num(grad_t)

                # Apply projection if specified
                if self.projector_kwargs is not None:
                    test_random_project = random_project(
                        grad_t,
                        grad_t.shape[0],
                        **self.projector_kwargs,
                    )
                    test_batch_grad = test_random_project(
                        grad_t,
                        ensemble_id=ckpt_idx,
                    )
                else:
                    test_batch_grad = grad_t

                # Apply normalization if specified
                if self.normalized_grad:
                    test_batch_grad = normalize(test_batch_grad)

                # Calculate influence scores for this batch
                col_st = test_batch_idx * test_dataloader.batch_size
                col_ed = min(
                    (test_batch_idx + 1) * test_dataloader.batch_size,
                    len(test_dataloader.sampler),
                )
                tda_output[:, col_st:col_ed] += (
                    (train_grads @ test_batch_grad.T * ckpt_weight)
                    .detach()
                    .cpu()
                )

        return tda_output

    def attribute_iterate(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Tensor:
        # Initialize output tensor
        if train_dataloader is not None:
            num_train = len(train_dataloader.sampler)
        else:
            num_train = len(self.full_train_dataloader.sampler)

        tda_output = torch.zeros(
            size=(num_train, len(test_dataloader.sampler)),
        )

        # iterate over each checkpoint (each ensemble)
        for ckpt_idx, ckpt_weight in zip(
            range(len(self.task.get_checkpoints())),
            self.weight_list,
        ):
            parameters, _ = self.task.get_param(
                ckpt_idx=ckpt_idx,
                layer_name=self.layer_name,
            )
            if self.layer_name is not None:
                self.grad_target_func = self.task.get_grad_target_func(
                    in_dims=(None, 0),
                    layer_name=self.layer_name,
                    ckpt_idx=ckpt_idx,
                )
                self.grad_loss_func = self.task.get_grad_loss_func(
                    in_dims=(None, 0),
                    layer_name=self.layer_name,
                    ckpt_idx=ckpt_idx,
                )

            for train_batch_idx, train_batch_data_ in enumerate(
                tqdm(
                    train_dataloader,
                    desc="calculating gradient of training set...",
                    leave=False,
                ),
            ):
                # Process training batch
                if isinstance(train_batch_data_, (tuple, list)):
                    train_batch_data = tuple(
                        data.to(self.device) for data in train_batch_data_
                    )
                else:
                    train_batch_data = train_batch_data_

                # get gradient of train
                grad_t = self.grad_loss_func(parameters, train_batch_data)
                grad_t = torch.nan_to_num(grad_t)
                if self.projector_kwargs is not None:
                    # define the projector for this batch of data
                    self.train_random_project = random_project(
                        grad_t,
                        grad_t.shape[0],
                        **self.projector_kwargs,
                    )
                    # param index as ensemble id
                    train_batch_grad = self.train_random_project(
                        grad_t,
                        ensemble_id=ckpt_idx,
                    )
                else:
                    train_batch_grad = grad_t

                for test_batch_idx, test_batch_data_ in enumerate(
                    tqdm(
                        test_dataloader,
                        desc="calculating gradient of test set...",
                        leave=False,
                    ),
                ):
                    # move to device
                    if isinstance(test_batch_data_, (tuple, list)):
                        test_batch_data = tuple(
                            data.to(self.device) for data in test_batch_data_
                        )
                    else:
                        test_batch_data = test_batch_data_

                    # get gradient of test
                    grad_t = self.grad_target_func(parameters, test_batch_data)
                    grad_t = torch.nan_to_num(grad_t)
                    if self.projector_kwargs is not None:
                        # define the projector for this batch of data
                        self.test_random_project = random_project(
                            grad_t,
                            grad_t.shape[0],
                            **self.projector_kwargs,
                        )

                        test_batch_grad = self.test_random_project(
                            grad_t,
                            ensemble_id=ckpt_idx,
                        )
                    else:
                        test_batch_grad = grad_t

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

        return tda_output

    def sparsity(
            self,
            data_loader: torch.utils.data.DataLoader,
            thresholds: List[float] = [0.1, 0.5, 0.9],
        ):
        if data_loader is None:
            raise ValueError("Please provide a dataloader to calculate sparsity.")

        sparsity = {threshold: {f"checkpoint_{i}": []
                    for i in range(len(self.task.get_checkpoints()))}
                    for threshold in thresholds}

        for ckpt_idx in range(len(self.task.get_checkpoints())):
            parameters, _ = self.task.get_param(
                ckpt_idx=ckpt_idx,
                layer_name=self.layer_name,
            )

            if self.layer_name is not None:
                self.grad_loss_func = self.task.get_grad_loss_func(
                    in_dims=(None, 0),
                    layer_name=self.layer_name,
                    ckpt_idx=ckpt_idx,
                )

            for batch in tqdm(
                data_loader,
                desc=f"calculating sparsity for checkpoint {ckpt_idx}...",
                leave=False,
            ):
                if isinstance(batch, (tuple, list)):
                    batch = tuple(data.to(self.device) for data in batch)
                else:
                    batch = batch

                grad_t = self.grad_loss_func(parameters, batch)
                grad_t = torch.nan_to_num(grad_t)

                if self.normalized_grad:
                    grad_t = normalize(grad_t)

                with torch.no_grad():
                    for threshold in thresholds:
                        grad_sparsity = (torch.abs(grad_t) < threshold).float().mean().item()
                        sparsity[threshold][f"checkpoint_{ckpt_idx}"].append(grad_sparsity)

        # Average sparsity across batches
        for threshold in thresholds:
            for ckpt_idx in range(len(self.task.get_checkpoints())):
                sparsity[threshold][f"checkpoint_{ckpt_idx}"] = \
                    sum(sparsity[threshold][f"checkpoint_{ckpt_idx}"]) / len(sparsity[threshold][f"checkpoint_{ckpt_idx}"])

        return sparsity