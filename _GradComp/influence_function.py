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
from .projector import setup_model_projectors

class IFAttributor:
    """
    Optimized influence function calculator using hooks for efficient gradient projection.
    Works with standard PyTorch layers.
    """

    def __init__(
        self,
        setting: str,
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
            setting (str): The setting of the experiment
            model (nn.Module): PyTorch model.
            layer_names (List[str]): Names of layers to attribute.
            hessian (str): Type of Hessian approximation ("none", "raw", "kfac", "ekfac"). Defaults to "raw".
            damping (float): Damping used when calculating the Hessian inverse. Defaults to None.
            profile (bool): Record time used in various parts of the algorithm run. Defaults to False.
            device (str): Device to run the model on. Defaults to 'cpu'.
            cpu_offload (bool): Whether to offload the model to CPU. Defaults to False.
            projector_kwargs (Dict): Keyword arguments for projector. Defaults to None.
        """
        self.setting = setting
        self.model = model
        self.model.to(device)
        self.model.eval()

        # Ensure layer_names is a list
        if isinstance(layer_names, str):
            self.layer_names = [layer_names]
        else:
            self.layer_names = layer_names

        self.hessian = hessian
        self.damping = damping
        self.profile = profile
        self.device = device
        self.cpu_offload = cpu_offload
        self.projector_kwargs = projector_kwargs or {}

        self.full_train_dataloader = None
        self.hook_manager = None
        self.cached_ifvp_train = None
        self.projectors = None

        # Initialize profiling stats
        if self.profile:
            self.profiling_stats = {
                'projection': 0.0,
                'forward': 0.0,
                'backward': 0.0,
                'precondition': 0.0,
            }

    def _setup_projectors(self, train_dataloader: torch.utils.data.DataLoader) -> None:
        """
        Set up projectors for the model layers

        Args:
            train_dataloader: DataLoader for training data
        """
        if not self.projector_kwargs:
            self.projectors = []
            return

        self.projectors = setup_model_projectors(
            self.model,
            self.layer_names,
            self.projector_kwargs,
            train_dataloader,
            self.setting,
            self.device
        )

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
        # Set up the projectors
        if self.projectors is None:
            self._setup_projectors(train_dataloader)

        # Create name-to-index mapping for layer access
        layer_name_to_idx = {name: idx for idx, name in enumerate(self.layer_names)}

        # Initialize dynamic lists to store gradients for each layer
        per_layer_gradients = [[] for _ in self.layer_names]

        # Create hook manager
        self.hook_manager = HookManager(
            self.model,
            self.layer_names,
        )

        # Set projectors in the hook manager
        if self.projectors:
            self.hook_manager.set_projectors(self.projectors)

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

            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['forward'] += time.time() - start_time

            # Compute custom loss
            logp = -outputs.loss
            train_loss = logp - torch.log(1 - torch.exp(logp))

            # Backward pass
            if self.profile:
                torch.cuda.synchronize(self.device)
                start_time = time.time()

            train_loss.backward()

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['backward'] += time.time() - start_time

            # Get projected gradients from hook manager
            with torch.no_grad():
                projected_grads = self.hook_manager.get_projected_grads()

                # Collect gradients on device first (no immediate CPU transfer)
                for idx, grad in enumerate(projected_grads):
                    if grad is not None:
                        # Keep gradients on GPU and append to the list
                        per_layer_gradients[idx].append(grad.detach())

        # Remove hooks after collecting all gradients
        self.hook_manager.remove_hooks()

        # Process all collected gradients after the loop
        hessians = []
        train_grads = []

        # Time for precondition calculations
        if self.profile:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Calculate Hessian and prepare gradients for each layer
        for layer_idx in range(len(self.layer_names)):
            if per_layer_gradients[layer_idx]:
                # Concatenate all batches for this layer on GPU
                grads = torch.cat(per_layer_gradients[layer_idx], dim=0)

                # Compute Hessian on GPU (more efficient)
                hessian = torch.matmul(grads.t(), grads) / len(train_dataloader.sampler)

                # Store results based on offload preference
                if self.cpu_offload:
                    # Move data to CPU only once (not in the batch loop)
                    train_grads.append(grads.cpu())
                    hessians.append(hessian.cpu())
                else:
                    train_grads.append(grads)
                    hessians.append(hessian)
            else:
                train_grads.append(None)
                hessians.append(None)

        if self.profile:
            torch.cuda.synchronize(self.device)
            self.profiling_stats['precondition'] += time.time() - start_time

        print(f"Computed gradient covariance for {len(self.layer_names)} modules")

        # Check if we have any valid gradients
        valid_grads = [grad is not None for grad in train_grads]
        if not any(valid_grads):
            print("Warning: No valid gradients were captured during the calculation.")
            print(f"Layer names: {self.layer_names}")
            return [torch.zeros(1) for _ in range(len(self.layer_names))]

        # Calculate inverse Hessian vector products
        if self.profile:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        if self.hessian == "none":
            return train_grads
        elif self.hessian == "raw":
            print("Computing gradient covariance inverse...")

            ifvp_train = []

            # Process each layer
            for layer_id, (grads, hessian) in enumerate(zip(train_grads, hessians)):
                if grads is None or hessian is None:
                    ifvp_train.append(None)
                    continue

                # Process Hessian inverse and IFVP calculation
                if self.cpu_offload:
                    # Move Hessian to GPU for inverse calculation
                    hessian_gpu = hessian.to(device=self.device)

                    # Calculate inverse on GPU (more efficient)
                    hessian_inv = stable_inverse(hessian_gpu, damping=self.damping)

                    # Process gradients in batches to avoid memory issues
                    batch_size = min(1024, grads.shape[0])  # Adjust based on available memory
                    results = []

                    for i in range(0, grads.shape[0], batch_size):
                        end_idx = min(i + batch_size, grads.shape[0])
                        grads_batch = grads[i:end_idx].to(device=self.device)

                        # Calculate IFVP for this batch
                        result_batch = torch.matmul(hessian_inv, grads_batch.t()).t()

                        # Move result back to CPU
                        results.append(result_batch.cpu())

                    # Combine results from all batches
                    ifvp_train.append(torch.cat(results, dim=0) if results else None)
                else:
                    # Calculate IFVP directly on GPU
                    hessian_inv = stable_inverse(hessian, damping=self.damping)
                    ifvp_train.append(torch.matmul(hessian_inv, grads.t()).t())

            print(f"Computed gradient covariance inverse for {len(self.layer_names)} modules")

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['precondition'] += time.time() - start_time

            return ifvp_train
        else:
            raise ValueError(f"Unsupported Hessian approximation: {self.hessian}")

    def cache(
        self,
        full_train_dataloader: torch.utils.data.DataLoader
    ) -> None:
        """
        Cache IFVP for the full training data.

        Args:
            full_train_dataloader: DataLoader for the full training data
        """
        print("Extracting information from training data...")
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

        # Set projectors in the hook manager if available
        if self.projectors:
            self.hook_manager.set_projectors(self.projectors)

        # Collect test gradients first (similar to training data approach)
        per_layer_test_gradients = [[] for _ in self.layer_names]
        test_batch_indices = []

        # Process each test batch
        print("Collecting projected gradients from test data...")
        for test_batch_idx, test_batch in enumerate(tqdm(test_dataloader, desc="Collecting test gradients")):
            # Zero gradients
            self.model.zero_grad()

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

                # Store test batch indices for later mapping
                batch_size = test_batch[0].shape[0] if not isinstance(test_batch, dict) else next(iter(test_batch.values())).shape[0]
                col_st = test_batch_idx * batch_size
                col_ed = min(col_st + batch_size, len(test_dataloader.sampler))
                test_batch_indices.append((col_st, col_ed))

                # Collect test gradients
                for idx, grad in enumerate(projected_grads):
                    if grad is not None:
                        per_layer_test_gradients[idx].append(grad.detach())

            # Zero gradients for next iteration
            self.model.zero_grad()

        # Remove hooks
        self.hook_manager.remove_hooks()

        # Process influence scores in batches based on collected test gradients
        batch_size = min(64, len(test_batch_indices))  # Process multiple test batches at once

        for batch_start in range(0, len(test_batch_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(test_batch_indices))

            # Calculate influence for each layer
            for layer_id in range(len(self.layer_names)):
                if not per_layer_test_gradients[layer_id]:
                    continue

                if ifvp_train[layer_id] is None:
                    continue

                # Process test gradients for this batch
                for batch_idx in range(batch_start, batch_end):
                    test_grad = per_layer_test_gradients[layer_id][batch_idx]
                    col_st, col_ed = test_batch_indices[batch_idx]

                    # Compute influence scores
                    if self.cpu_offload:
                        # Move data to GPU in batches
                        ifvp_batch_size = min(1024, ifvp_train[layer_id].shape[0])

                        for i in range(0, ifvp_train[layer_id].shape[0], ifvp_batch_size):
                            end_idx = min(i + ifvp_batch_size, ifvp_train[layer_id].shape[0])

                            ifvp_batch = ifvp_train[layer_id][i:end_idx].to(device=self.device)
                            test_grad_gpu = test_grad.to(device=self.device)

                            # Compute partial influence
                            result = torch.matmul(ifvp_batch, test_grad_gpu.t())

                            # Update influence scores
                            IF_score_segment = IF_score[i:end_idx, col_st:col_ed].to(device=self.device)
                            IF_score_segment += result
                            IF_score[i:end_idx, col_st:col_ed] = IF_score_segment.cpu()
                    else:
                        # Compute on GPU directly
                        result = torch.matmul(ifvp_train[layer_id], test_grad.t())
                        IF_score[:, col_st:col_ed] += result

        # Return result
        if self.profile:
            return (IF_score, self.profiling_stats)
        else:
            return IF_score