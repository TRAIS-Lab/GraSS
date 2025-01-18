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
        # these are projector kwargs shared by train/test projector
        self.projector_kwargs = projector_kwargs
        # set proj seed
        if projector_kwargs is not None:
            self.threshold = self.projector_kwargs.get("threshold", None)
            self.projector_kwargs.pop("threshold")

        self.normalized_grad = normalized_grad
        self.layer_name = layer_name
        self.device = device
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
            print(parameters.shape)
            if self.layer_name is not None:
                self.grad_loss_func = self.task.get_grad_loss_func(
                    in_dims=(None, 0),
                    layer_name=self.layer_name,
                    ckpt_idx=ckpt_idx,
                )

            train_batch_grads = []
            for train_data in tqdm(
                full_train_dataloader,
                desc=f"caching gradients for checkpoint {ckpt_idx}...",
                leave=False,
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
                # Apply thresholding if specified
                if self.projector_kwargs is not None and self.threshold is not None:
                    grad_t = torch.where(
                        grad_t.abs() > self.threshold,
                        grad_t,
                        torch.zeros_like(grad_t),
                    )

                # Apply projection if specified
                if self.projector_kwargs is not None:
                    train_random_project = random_project(
                        grad_t,
                        grad_t.shape[0],
                        **self.projector_kwargs,
                    )
                    grad_t = train_random_project(
                        torch.nan_to_num(grad_t),
                        ensemble_id=ckpt_idx,
                    )
                else:
                    grad_t = torch.nan_to_num(grad_t)

                # Apply normalization if specified
                if self.normalized_grad:
                    grad_t = normalize(grad_t)

                train_batch_grads.append(grad_t.detach())

            # Store concatenated gradients for this checkpoint
            self.cached_train_grads.append(torch.cat(train_batch_grads, dim=0))

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
            raise ValueError(
                "Cannot provide train_dataloader when cached gradients exist. "
                "Either use cached gradients (train_dataloader=None) or clear the cache."
            )

        # Check checkpoint and weight list lengths match
        if len(self.task.get_checkpoints()) != len(self.weight_list):
            raise ValueError("The length of checkpoints and weights lists don't match.")

        # Initialize output tensor
        if train_dataloader is not None:
            num_train = len(train_dataloader.sampler)
        else:
            num_train = len(self.full_train_dataloader.sampler)

        tda_output = torch.zeros(
            size=(num_train, len(test_dataloader.sampler)),
        )

        # Iterate over checkpoints
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

            # Handle training gradients - either compute or use cached
            if train_dataloader is not None:
                train_grads = []
                for train_batch_data_ in tqdm(
                    train_dataloader,
                    desc="calculating gradient of training set...",
                    leave=False,
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

                    # Apply thresholding if specified
                    if self.projector_kwargs is not None and self.threshold is not None:
                        grad_t = torch.where(
                            grad_t.abs() > self.threshold,
                            grad_t,
                            torch.zeros_like(grad_t),
                        )

                    # Apply projection if specified
                    if self.projector_kwargs is not None:
                        train_random_project = random_project(
                            grad_t,
                            grad_t.shape[0],
                            **self.projector_kwargs,
                        )
                        grad_t = train_random_project(
                            torch.nan_to_num(grad_t),
                            ensemble_id=ckpt_idx,
                        )
                    else:
                        grad_t = torch.nan_to_num(grad_t)

                    # Apply normalization if specified
                    if self.normalized_grad:
                        grad_t = normalize(grad_t)

                    train_grads.append(grad_t)

                train_grads = torch.cat(train_grads, dim=0)
            else:
                # Use cached gradients
                train_grads = self.cached_train_grads[ckpt_idx].to(self.device)

            # Process test data batches
            curr_col = 0
            for test_batch_data_ in tqdm(
                test_dataloader,
                desc="calculating gradient of test set...",
                leave=False,
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

                # Apply thresholding if specified
                if self.projector_kwargs is not None and self.threshold is not None:
                    grad_t = torch.where(
                        grad_t.abs() > self.threshold,
                        grad_t,
                        torch.zeros_like(grad_t),
                    )

                # Apply projection if specified
                if self.projector_kwargs is not None:
                    test_random_project = random_project(
                        grad_t,
                        grad_t.shape[0],
                        **self.projector_kwargs,
                    )
                    test_batch_grad = test_random_project(
                        torch.nan_to_num(grad_t),
                        ensemble_id=ckpt_idx,
                    )
                else:
                    test_batch_grad = torch.nan_to_num(grad_t)

                # Apply normalization if specified
                if self.normalized_grad:
                    test_batch_grad = normalize(test_batch_grad)

                # Calculate influence scores for this batch
                batch_size = test_batch_grad.shape[0]
                tda_output[:, curr_col:curr_col + batch_size] += (
                    (train_grads @ test_batch_grad.T * ckpt_weight)
                    .detach()
                    .cpu()
                )
                curr_col += batch_size

        return tda_output
