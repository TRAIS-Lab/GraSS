from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
import time

from .hook import HookManager
from .utils import stable_inverse

class IFAttributor:
    """
    Optimized influence function calculator using hooks for efficient gradient projection.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: Union[str, List[str]],
        hessian: str = "raw",
        damping: float = None,
        profile: bool = False,
        device: str = 'cpu',
        cpu_offload: bool = False,
        projector_kwargs: Dict = None,
    ) -> None:
        """
        Optimized Influence Function Attributor.

        Args:
            model (nn.Module): PyTorch model.
            layer_names (List[str]): Names of layers to attribute.
            hessian (str): Type of Hessian approximation ("none", "raw", "kfac", "ekfac"). Defaults to "raw".
            damping (float): Damping used when calculating the Hessian inverse. Defaults to None.
            profile (bool): Record time used in various parts of the algorithm run. Defaults to False.
            device (str): Device to run the model on. Defaults to 'cpu'.
            cpu_offload (bool): Whether to offload the model to CPU. Defaults to False.
            projector_kwargs (Dict): Keyword arguments for projector. Defaults to None.
        """
        self.model = model
        self.model.to(device)
        self.model.eval()

        self.layer_names = layer_names
        self.hessian = hessian  # Currently only raw is supported
        self.damping = damping
        self.profile = profile
        self.device = device
        self.cpu_offload = cpu_offload
        self.projector_kwargs = projector_kwargs or {}

        self.full_train_dataloader = None
        self.hook_manager = None

        # Initialize profiling stats
        if self.profile:
            self.profiling_stats = {
                'projection': 0.0,
                'forward': 0.0,
                'backward': 0.0,
                'precondition': 0.0,
            }

    def _calculate_ifvp(
        self,
        train_dataloader: torch.utils.data.DataLoader,
    ) -> List[torch.Tensor]:
        """
        Compute FIM inverse Hessian vector product for each layer using hooks.

        Args:
            train_dataloader: DataLoader for training data

        Returns:
            List of tensors containing the FIM factors for each layer
        """
        # Set up the projector
        self.model.set_projectors(self.layer_names, self.projector_kwargs, train_dataloader)

        num_layers = len(self.layer_names)
        num_samples = len(train_dataloader.sampler)

        # Initialize Hessian on the appropriate device
        Hessian = [torch.zeros((1, 1), device="cpu" if self.cpu_offload else self.device) for _ in range(num_layers)]

        # Initialize train gradients
        train_grads = [None] * num_layers

        # Create hook manager
        self.hook_manager = HookManager(
            self.model,
            self.layer_names,
        )

        # Iterate through the training data to compute gradients
        for train_batch_idx, train_batch in enumerate(tqdm(train_dataloader, desc="Processing training data")):
            # Zero gradients
            self.model.zero_grad()

            # Prepare inputs
            if isinstance(train_batch, dict):
                inputs = {k: v.to(self.device) for k, v in train_batch.items()}
            else:
                inputs = train_batch[0].to(self.device)

            # Forward pass
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
            train_loss.backward()

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['backward'] += time.time() - start_time

            # Get projected gradients from hook manager
            with torch.no_grad():
                projected_grads = self.hook_manager.get_projected_grads()
                for layer_name, grad in projected_grads.items():
                    # Find corresponding layer_id
                    layer_id = self.layer_names.index(layer_name)

                    # Initialize train_grads if not done yet
                    if train_grads[layer_id] is None:
                        storage_device = "cpu" if self.cpu_offload else self.device
                        train_grads[layer_id] = torch.zeros((num_samples, *grad.shape[1:]), device=storage_device)

                    # Store gradients
                    col_st = train_batch_idx * train_dataloader.batch_size
                    col_ed = min(
                        (train_batch_idx + 1) * train_dataloader.batch_size,
                        num_samples,
                    )
                    if self.cpu_offload:
                        train_grads[layer_id][col_st:col_ed] = grad.detach().cpu()
                    else:
                        train_grads[layer_id][col_st:col_ed] = grad.detach()

                    # Compute Hessian
                    if self.profile:
                        torch.cuda.synchronize(self.device)
                        start_time = time.time()

                    if self.cpu_offload:
                        if Hessian[layer_id].shape[0] == 1:  # First iteration
                            Hessian[layer_id] = torch.zeros((grad.shape[1], grad.shape[1]), device="cpu")
                        grad_gpu = grad.to(device=self.device)
                        hessian_update = torch.matmul(grad_gpu.t(), grad_gpu) / num_samples
                        Hessian[layer_id] += hessian_update.cpu()
                    else:
                        if Hessian[layer_id].shape[0] == 1:  # First iteration
                            Hessian[layer_id] = torch.zeros((grad.shape[1], grad.shape[1]), device=self.device)
                        Hessian[layer_id] += torch.matmul(grad.t(), grad) / num_samples

                    if self.profile:
                        torch.cuda.synchronize(self.device)
                        self.profiling_stats['precondition'] += time.time() - start_time

        # Remove hooks
        self.hook_manager.remove_hooks()

        # Check if we have any valid gradients
        valid_grads = [grad is not None for grad in train_grads]
        if not any(valid_grads):
            print("Warning: No valid gradients were captured during the calculation.")
            print(f"Layer names: {self.layer_names}")
            return [torch.zeros(1) for _ in range(num_layers)]

        # Calculate inverse Hessian vector products
        if self.profile:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        if self.hessian == "none":
            return train_grads
        elif self.hessian == "raw":
            ifvp_train = [None] * num_layers

            # Add damping term and compute inverses
            for layer_id in range(num_layers):
                if train_grads[layer_id] is None:
                    continue

                if self.cpu_offload:
                    hessian_gpu = Hessian[layer_id].to(device=self.device)
                    train_grad_gpu = train_grads[layer_id].to(device=self.device)

                    hessian_inv = stable_inverse(hessian_gpu, damping=self.damping)
                    result = torch.matmul(hessian_inv, train_grad_gpu.t()).t()

                    ifvp_train[layer_id] = result.cpu()
                else:
                    ifvp_train[layer_id] = torch.matmul(
                        stable_inverse(Hessian[layer_id], damping=self.damping),
                        train_grads[layer_id].t()
                    ).t()

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['precondition'] += time.time() - start_time

            return ifvp_train

    def cache(
        self,
        full_train_dataloader: torch.utils.data.DataLoader
    ) -> None:
        """
        Cache IFVP for the full training data.

        Args:
            full_train_dataloader: DataLoader for the full training data
        """
        self.full_train_dataloader = full_train_dataloader
        self.cached_ifvp_train = self._calculate_ifvp(full_train_dataloader)

    def attribute(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Attribute influence of training examples on test examples.

        Args:
            test_dataloader: DataLoader for test data
            train_dataloader: Optional DataLoader for training data if not cached

        Returns:
            Tensor of influence scores (and profiling stats if profile=True)
        """
        if self.full_train_dataloader is not None and train_dataloader is not None:
            raise ValueError(
                "You have cached a training loader by .cache() and you are trying to attribute "
                "a different training loader. If this new training loader is a subset of the cached "
                "training loader, please don't input the training dataloader in .attribute() and "
                "directly use index to select the corresponding scores."
            )

        if train_dataloader is None and self.full_train_dataloader is None:
            raise ValueError(
                "You did not state a training loader in .attribute() and you did not cache a "
                "training loader by .cache(). Please provide a training loader or cache a "
                "training loader."
            )

        # Use cached IFVP or calculate new ones
        if train_dataloader is not None and self.full_train_dataloader is None:
            num_train = len(train_dataloader.sampler)
            ifvp_train = self._calculate_ifvp(train_dataloader)
        else:
            num_train = len(self.full_train_dataloader.sampler)
            ifvp_train = self.cached_ifvp_train

        # Storage device
        storage_device = "cpu" if self.cpu_offload else self.device

        # Initialize influence scores
        IF_score = torch.zeros(num_train, len(test_dataloader.sampler), device=storage_device)

        # Create hook manager for test examples
        self.hook_manager = HookManager(
            self.model,
            self.layer_names,
        )

        # Compute influence scores
        for test_batch_idx, test_batch in enumerate(tqdm(test_dataloader, desc="Processing test data")):
            # Prepare inputs
            if isinstance(test_batch, dict):
                inputs = {k: v.to(self.device) for k, v in test_batch.items()}
            else:
                inputs = test_batch[0].to(self.device)

            # Forward pass
            if self.profile:
                torch.cuda.synchronize(self.device)
                start_time = time.time()

            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['forward'] += time.time() - start_time

            # Compute loss
            logp = -outputs.loss
            test_loss = logp - torch.log(1 - torch.exp(logp))

            # Backward pass
            if self.profile:
                torch.cuda.synchronize(self.device)
                start_time = time.time()

            test_loss.backward()

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['backward'] += time.time() - start_time

            # Get projected gradients from hook manager
            with torch.no_grad():
                projected_grads = self.hook_manager.get_projected_grads()

                for layer_id, layer_name in enumerate(self.layer_names):
                    if layer_name in projected_grads:
                        test_grad = projected_grads[layer_name]

                        col_st = test_batch_idx * test_dataloader.batch_size
                        col_ed = min(
                            (test_batch_idx + 1) * test_dataloader.batch_size,
                            len(test_dataloader.sampler),
                        )

                        # Compute influence scores
                        if self.cpu_offload:
                            ifvp_train_gpu = ifvp_train[layer_id].to(device=self.device)
                            test_grad_gpu = test_grad.to(device=self.device)

                            result = torch.matmul(ifvp_train_gpu, test_grad_gpu.t())

                            IF_score_segment = IF_score[:, col_st:col_ed].to(device=self.device)
                            IF_score_segment += result
                            IF_score[:, col_st:col_ed] = IF_score_segment.cpu()
                        else:
                            result = torch.matmul(ifvp_train[layer_id], test_grad.t())
                            IF_score[:, col_st:col_ed] += result

            # Zero gradients
            self.model.zero_grad()

        # Remove hooks
        self.hook_manager.remove_hooks()

        # Return result
        if self.profile:
            return (IF_score, self.profiling_stats)
        else:
            return IF_score