from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
import time

from .utils import stable_inverse

class IFAttributor:
    """
    Influence function calculator that uses vanilla gradient components.

    This class implements efficient influence function calculation using
    customized projection to reduce the dimensionality of gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_name: Optional[Union[str, List[str]]],
        hessian: str = "raw",
        damping: float = None,
        profile: bool = False,
        device: str = 'cpu',
        cpu_offload: bool = False,
    ) -> None:
        """
        Influence Function Attributor equivalent to LoGra.

        Args:
            model (nn.Module): PyTorch model.
            layer_name (Optional[Union[str, List[str]]], optional): All layers the are going to be attribute. Defaults to None.
            hessian (str): Type of Hessian approximation hessian ("none", "raw", "kfac", "ekfac"). Defaults to "raw".
            damping (float): Damping used when calculating the Hessian inverse. Defaults to None.
            profile (bool): Record time used in various parts of the algorithm run. Defaults to False.
            device (str): Device to run the model on. Defaults to 'cpu'.
            cpu_offload (bool): Whether to offload the model to CPU. Defaults to False.
        """
        self.model = model
        self.layer_name = layer_name
        self.hessian = hessian # Currently only raw is supported
        self.damping = damping
        self.profile = profile
        self.device = device
        self.cpu_offload = cpu_offload

        self.full_train_dataloader = None

        # Initialize profiling stats
        if self.profile:
            self.profiling_stats = {
                'projection': 0.0,
                'forward': 0.0,
                'backward': 0.0,
                'precondition': 0.0,
            }

    def _iFVP(
        self,
        train_dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute FIM inverse Hessian vector product for each layer.
        Uses GCN approximation for the Hessian computation.

        Args:
            train_dataloader: DataLoader for training data
            cached_grads: Whether gradients are cached

        Returns:
            Tuple of lists containing the FIM factors for each layer
        """
        num_layers = len(self.layer_name)

        # Initialize Hessian on the appropriate device
        Hessian = [torch.zeros((1, 1), device="cpu" if self.cpu_offload else self.device) for _ in range(num_layers)]

        train_grad = [None] * len(self.layer_name)

        num_samples = len(train_dataloader.sampler)

        # Compute FIM factors for each layer
        for train_batch_idx, train_batch in enumerate(
            tqdm(
                train_dataloader,
                desc="computing FIM factors for training set...",
                leave=False,
            ),
        ):
            # Prepare inputs
            if isinstance(train_batch, dict):
                inputs = {k: v.to(self.model.device) for k, v in train_batch.items()}
            else:
                inputs = train_batch[0].to(self.model.device)

            # Time forward pass
            if self.profile:
                torch.cuda.synchronize(self.device)
                start_time = time.time()

            # Forward pass
            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['forward'] += time.time() - start_time

            # Compute custom loss
            logp = -outputs.loss
            train_loss = logp - torch.log(1 - torch.exp(logp))

            # Time backward pass
            if self.profile:
                torch.cuda.synchronize(self.device)
                start_time = time.time()

            # Backward pass
            train_pre_acts = [layer.pre_activation for layer in self.layer_name]
            Z_grad_train = torch.autograd.grad(train_loss, train_pre_acts, retain_graph=True)

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['backward'] += time.time() - start_time

            # Compute FIM factors for each layer
            with torch.no_grad():
                for layer_id, (layer, z_grad_train) in enumerate(zip(self.layer_name, Z_grad_train)):
                    if self.profile:
                        torch.cuda.synchronize(self.device)
                        start_time = time.time()

                    # Calculate gradient
                    grad = layer.grad_from_grad_comp(z_grad_train, per_sample=True)

                    if self.profile:
                        torch.cuda.synchronize(self.device)
                        self.profiling_stats['projection'] += time.time() - start_time

                    # Initialize train_grad if not done yet
                    if train_grad[layer_id] is None:
                        # Choose device based on cpu_offload setting
                        storage_device = "cpu" if self.cpu_offload else self.device
                        train_grad[layer_id] = torch.zeros((num_samples, *grad.shape[1:]), device=storage_device)

                    col_st = train_batch_idx * train_dataloader.batch_size
                    col_ed = min(
                        (train_batch_idx + 1) * train_dataloader.batch_size,
                        len(train_dataloader.sampler),
                    )

                    # Store gradients with CPU offloading if enabled
                    if self.cpu_offload:
                        train_grad[layer_id][col_st:col_ed] = grad.detach().cpu()
                    else:
                        train_grad[layer_id][col_st:col_ed] = grad.detach()

                    # Time Hessian calculation
                    if self.profile:
                        torch.cuda.synchronize(self.device)
                        start_time = time.time()

                    # Compute Hessian with CPU offloading if enabled
                    if self.cpu_offload:
                        # Ensure Hessian is properly sized
                        if Hessian[layer_id].shape[0] == 1:  # First iteration
                            Hessian[layer_id] = torch.zeros((grad.shape[1], grad.shape[1]), device="cpu")

                        # Move to GPU for computation, then back to CPU
                        grad_gpu = grad.to(device=self.device)
                        hessian_update = torch.matmul(grad_gpu.t(), grad_gpu) / num_samples
                        Hessian[layer_id] += hessian_update.cpu()
                    else:
                        # Ensure Hessian is properly sized
                        if Hessian[layer_id].shape[0] == 1:  # First iteration
                            Hessian[layer_id] = torch.zeros((grad.shape[1], grad.shape[1]), device=self.device)

                        Hessian[layer_id] += torch.matmul(grad.t(), grad) / num_samples

                    if self.profile:
                        torch.cuda.synchronize(self.device)
                        self.profiling_stats['precondition'] += time.time() - start_time

        # Time inverse Hessian calculation
        if self.profile:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        if self.hessian == "none":
            # If no Hessian approximation is used, return None
            return train_grad
        elif self.hessian == "raw":
            print(f"Calculating iFVP...")
            ifvp_train = [None] * len(self.layer_name)
            # Add damping term and compute inverses
            for layer_id in range(num_layers):
                if self.cpu_offload:
                    # Move to GPU for computation
                    hessian_gpu = Hessian[layer_id].to(device=self.device)
                    train_grad_gpu = train_grad[layer_id].to(device=self.device)

                    # Compute inverse and product
                    hessian_inv = stable_inverse(hessian_gpu, damping=self.damping)
                    result = torch.matmul(hessian_inv, train_grad_gpu.t()).t()

                    # Move back to CPU
                    ifvp_train[layer_id] = result.cpu()
                else:
                    ifvp_train[layer_id] = torch.matmul(
                        stable_inverse(Hessian[layer_id], damping=self.damping),
                        train_grad[layer_id].t()
                    ).t()

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['precondition'] += time.time() - start_time

            return ifvp_train

    def cache(
        self,
        full_train_dataloader: torch.utils.data.DataLoader
    ) -> None:
        # This means we can afford full calculation.
        self.full_train_dataloader = full_train_dataloader
        self.cached_ifvp_train = self._iFVP(full_train_dataloader)

    def attribute(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Attributing the test set with respect to the training set.

        Args:
            test_dataloader (torch.utils.data.DataLoader): _description_
            train_dataloader (Optional[torch.utils.data.DataLoader], optional): _description_. Defaults to None.

        Returns:
            If profile=False:
                torch.Tensor: The influence scores
            If profile=True:
                Tuple[torch.Tensor, Dict]: The influence scores and profiling statistics
        """
        if self.full_train_dataloader is not None and train_dataloader is not None: # if cached
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

        return self.attribute_default(test_dataloader, train_dataloader)

    def attribute_default(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> torch.Tensor:
        """
        Compute influence scores using FIM approximation.

        Args:
            test_dataloader: DataLoader for test data
            train_dataloader: Optional DataLoader for training data if not cached

        Returns:
            Tensor of influence scores
        """
        if train_dataloader is not None:
            num_train = len(train_dataloader.sampler)
        else:
            num_train = len(self.full_train_dataloader.sampler)

        # Initialize IF_score on appropriate device based on cpu_offload
        storage_device = "cpu" if self.cpu_offload else self.device
        IF_score = torch.zeros(num_train, len(test_dataloader.sampler), device=storage_device)

        # Compute FIM factors if not cached
        if train_dataloader is not None and self.full_train_dataloader is None:
            ifvp_train = self._iFVP(train_dataloader)
        else:
            ifvp_train = self.cached_ifvp_train

        # Compute influence scores
        for test_batch_idx, test_batch in enumerate(
            tqdm(test_dataloader, desc="computing influence scores...", leave=False),
        ):
            # Prepare inputs
            if isinstance(test_batch, dict):
                inputs = {k: v.to(self.model.device) for k, v in test_batch.items()}
            else:
                inputs = test_batch[0].to(self.model.device)

             # Time forward pass
            if self.profile:
                torch.cuda.synchronize(self.device)
                start_time = time.time()

            # Forward pass
            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['forward'] += time.time() - start_time

            #Compute custom loss
            logp = -outputs.loss
            test_loss = logp - torch.log(1 - torch.exp(logp))

            # Time backward pass
            if self.profile:
                torch.cuda.synchronize(self.device)
                start_time = time.time()

            # Backward pass
            test_pre_acts = [layer.pre_activation for layer in self.layer_name]
            Z_grad_test = torch.autograd.grad(test_loss, test_pre_acts, retain_graph=True)

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['backward'] += time.time() - start_time

            with torch.no_grad():
                for layer_id, (layer, z_grad_test) in enumerate(zip(self.layer_name, Z_grad_test)):
                    if self.profile:
                        torch.cuda.synchronize(self.device)
                        start_time = time.time()

                    test_grad = layer.grad_from_grad_comp(z_grad_test, per_sample=True)

                    if self.profile:
                        torch.cuda.synchronize(self.device)
                        self.profiling_stats['projection'] += time.time() - start_time

                    col_st = test_batch_idx * test_dataloader.batch_size
                    col_ed = min(
                        (test_batch_idx + 1) * test_dataloader.batch_size,
                        len(test_dataloader.sampler),
                    )

                    # Compute the influence function score with CPU offloading if enabled
                    if self.cpu_offload:
                        # Move data to GPU for computation
                        ifvp_train_gpu = ifvp_train[layer_id].to(device=self.device)
                        test_grad_gpu = test_grad.to(device=self.device)

                        # Compute on GPU
                        result = torch.matmul(ifvp_train_gpu, test_grad_gpu.t())

                        # Update IF_score on CPU
                        IF_score_segment = IF_score[:, col_st:col_ed].to(device=self.device)
                        IF_score_segment += result
                        IF_score[:, col_st:col_ed] = IF_score_segment.cpu()
                    else:
                        # Compute directly on the current device
                        result = torch.matmul(ifvp_train[layer_id], test_grad.t())
                        IF_score[:, col_st:col_ed] += result

        # Move final result to desired device if needed
        final_result = IF_score
        if self.profile:
            return (final_result, self.profiling_stats)
        else:
            return final_result