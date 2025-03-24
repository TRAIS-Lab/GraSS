from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import List, Optional, Union, Tuple

import torch
from tqdm import tqdm

import time

def stable_inverse(matrix: torch.Tensor, damping: float = None) -> torch.Tensor:
    """
    Compute a numerically stable inverse of a matrix using eigendecomposition.

    Args:
        matrix: Input matrix to invert
        damping: Damping factor for numerical stability

    Returns:
        Stable inverse of the input matrix
    """
    # sometimes the matrix is a single number, so we need to check if it's a scalar
    if len(matrix.shape) == 0:
        if matrix == 0:
            # return a 2d 0 tensor
            return torch.tensor([[0.0]], device=matrix.device)
        else:
            return torch.tensor([[1.0 / (matrix * 1.1)]], device=matrix.device)

    # Add damping to the diagonal
    if damping is None:
        damping = 0.1 * torch.trace(matrix) / matrix.size(0)

    damped_matrix = matrix + damping * torch.eye(matrix.size(0), device=matrix.device)

    try:
        # Try Cholesky decomposition first (more stable)
        L = torch.linalg.cholesky(damped_matrix)
        inverse = torch.cholesky_inverse(L)
    except RuntimeError:
        print(f"Falling back to direct inverse due to Cholesky failure")
        # Fall back to direct inverse
        inverse = torch.inverse(damped_matrix)

    return inverse

class GCIFAttributorRAW():
    def __init__(
        self,
        model,
        layer_name: Optional[Union[str, List[str]]],
        damping = None,
        profile: bool = False,
        device: str = 'cpu'
    ) -> None:
        """Ghost Inner Product Attributor for Gradient Dot.

        Args:
            model (_type_): _description_
            layer_name (Optional[Union[str, List[str]]], optional): _description_. Defaults to None.
            projector_kwargs (Optional[Dict[str, Any]], optional): _description_. Defaults to None.
            damping (float): Damping used when calculating the Hessian inverse. Defaults to 1e-4.
            profile (bool, optional): Record time used in various parts of the algorithm run. Defaults to False.
            mode (str, optional): Currently only "default" mode is supported. Defaults to "default".
            device (str, optional): _description_. Defaults to 'cpu'.
        """
        self.model = model
        self.layer_name = layer_name
        self.damping = damping
        self.profile = profile
        self.device = device
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

        Hessian = [0] * num_layers

        train_grad = [None] * len(self.layer_name)
        ihvp_train = [None] * len(self.layer_name)

        num_samples = len(train_dataloader.sampler)

        # Compute FIM factors for each layer
        for train_batch_idx, train_batch in enumerate(
            tqdm(
                train_dataloader,
                desc="computing FIM factors for training set...",
                leave=False,
            ),
        ):
            train_input_ids = train_batch["input_ids"].to(self.device)
            train_attention_masks = train_batch["attention_mask"].to(self.device)
            train_labels = train_batch["labels"].to(self.device)

            # Time forward/backward pass
            if self.profile:
                torch.cuda.synchronize()
                start_time = time.time()

            # Forward pass
            #TODO: update to {k: v.to(self.device) for k, v in train_batch.items()}
            outputs = self.model(
                input_ids=train_input_ids,
                attention_mask=train_attention_masks,
                labels=train_labels
            )

            if self.profile:
                torch.cuda.synchronize()
                self.profiling_stats['forward'] += time.time() - start_time

            logp = -outputs.loss
            train_loss = logp - torch.log(1 - torch.exp(logp))

            if self.profile:
                torch.cuda.synchronize()
                start_time = time.time()

            train_pre_acts = [layer.pre_activation for layer in self.layer_name]
            Z_grad_train = torch.autograd.grad(train_loss, train_pre_acts, retain_graph=True)

            if self.profile:
                torch.cuda.synchronize()
                self.profiling_stats['backward'] += time.time() - start_time

            # Compute FIM factors for each layer
            with torch.no_grad():
                for layer_id, (layer, z_grad_full) in enumerate(zip(self.layer_name, Z_grad_train)):
                    if self.profile:
                        torch.cuda.synchronize()
                        start_time = time.time()

                    grad_comp_1, grad_comp_2 = layer.grad_comp(z_grad_full, per_sample=True)
                    grad = layer.grad_from_grad_comp(grad_comp_1, grad_comp_2)

                    if self.profile:
                        torch.cuda.synchronize()
                        self.profiling_stats['projection'] += time.time() - start_time

                    if train_grad[layer_id] is None:
                        train_grad[layer_id] = torch.zeros((num_samples, *grad.shape[1:]), device=self.device)

                    col_st = train_batch_idx * train_dataloader.batch_size
                    col_ed = min(
                        (train_batch_idx + 1) * train_dataloader.batch_size,
                        len(train_dataloader.sampler),
                    )
                    train_grad[layer_id][col_st:col_ed] = grad.detach()

                    # Time Hessian calculation
                    if self.profile:
                        torch.cuda.synchronize()
                        start_time = time.time()

                    Hessian[layer_id] += torch.matmul(grad.t(), grad) / num_samples

                    if self.profile:
                        torch.cuda.synchronize()
                        self.profiling_stats['precondition'] += time.time() - start_time

        # Time inverse Hessian calculation
        if self.profile:
            torch.cuda.synchronize()
            start_time = time.time()

        print(f"Calculating iHVP...")
        # Add damping term and compute inverses
        for layer_id in range(num_layers):
            ihvp_train[layer_id] = torch.matmul(
                stable_inverse(Hessian[layer_id], damping=self.damping),
                train_grad[layer_id].t()
            ).t()

        if self.profile:
            torch.cuda.synchronize()
            self.profiling_stats['precondition'] += time.time() - start_time

        return ihvp_train

    def cache(
        self,
        full_train_dataloader: torch.utils.data.DataLoader
    ) -> None:
        # This means we can afford full calculation.
        self.full_train_dataloader = full_train_dataloader
        self.cached_ihvp_train = self._iFVP(full_train_dataloader)

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

        IF_score = torch.zeros(num_train, len(test_dataloader.sampler), device=self.device)

        # Compute FIM factors if not cached
        if train_dataloader is not None and self.full_train_dataloader is None:
            ihvp_train = self._iFVP(train_dataloader)
        else:
            ihvp_train = self.cached_ihvp_train

        # Compute influence scores
        for test_batch_idx, test_batch in enumerate(
            tqdm(test_dataloader, desc="computing influence scores...", leave=False),
        ):
            test_input_ids = test_batch["input_ids"].to(self.device)
            test_attention_masks = test_batch["attention_mask"].to(self.device)
            test_labels = test_batch["labels"].to(self.device)

             # Time forward pass
            if self.profile:
                torch.cuda.synchronize()
                start_time = time.time()

            outputs = self.model(
                input_ids=test_input_ids,
                attention_mask=test_attention_masks,
                labels=test_labels
            )

            if self.profile:
                torch.cuda.synchronize()
                self.profiling_stats['forward'] += time.time() - start_time

            logp = -outputs.loss
            test_loss = logp - torch.log(1 - torch.exp(logp))

            # Time backward pass
            if self.profile:
                torch.cuda.synchronize()
                start_time = time.time()

            test_pre_acts = [layer.pre_activation for layer in self.layer_name]
            Z_grad_test = torch.autograd.grad(test_loss, test_pre_acts, retain_graph=True)

            if self.profile:
                torch.cuda.synchronize()
                self.profiling_stats['backward'] += time.time() - start_time

            with torch.no_grad():
                for layer_id, (layer, z_grad_test) in enumerate(zip(self.layer_name, Z_grad_test)):
                    if self.profile:
                        torch.cuda.synchronize()
                        start_time = time.time()

                    grad_comp_1, grad_comp_2 = layer.grad_comp(z_grad_test, per_sample=True)
                    test_grad = layer.grad_from_grad_comp(grad_comp_1, grad_comp_2)

                    if self.profile:
                        torch.cuda.synchronize()
                        self.profiling_stats['projection'] += time.time() - start_time


                    col_st = test_batch_idx * test_dataloader.batch_size
                    col_ed = min(
                        (test_batch_idx + 1) * test_dataloader.batch_size,
                        len(test_dataloader.sampler),
                    )

                    # Compute the influence function score from pre-computed iHVP components and test gradients
                    result = torch.matmul(ihvp_train[layer_id], test_grad.t())

                    IF_score[:, col_st:col_ed] += result

        return (IF_score, self.profiling_stats) if self.profile else IF_score