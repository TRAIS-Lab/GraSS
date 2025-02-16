from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import List, Optional, Union, Tuple

import torch
from torch import Tensor
from tqdm import tqdm
from .layers.linear import GIPLinear, GIPEmbedding
from .layers.layer_norm import GIPLayerNorm
from .helper import find_GIPlayers

from _dattri.func.projection import random_project

import time

def setup_projectors(
        projector_kwargs: Dict[str, Any],
        layer_name: List[Union[GIPLinear, GIPEmbedding, GIPLayerNorm]],
        mode: Optional[str] = "default",
    ) -> Tuple[Dict[str, Any], List[int], int]:
    """Setup projection dimensions and seeds for each layer.

    Args:
        projector_kwargs (Dict[str, Any]): projector's arguments.
        layer_name (List[Union[GIPLinear, GIPEmbedding, GIPLayerNorm]]): the list of layers to be projected.
        mode (Optional[str], optional): the data attribution's running mode. Defaults to "default".

    Returns:
        Tuple[Dict[str, Any], List[int], int]: processed projector's arguments, projection dimensions, and projection seed.
    """
    if projector_kwargs is None:
        return None, None, None
    proj_seed = projector_kwargs.get("proj_seed", 0)
    proj_dim = projector_kwargs.get("proj_dim", 512)
    proj_dim_dist = projector_kwargs.get("proj_dim_dist", "uniform")

    projector_kwargs.pop("proj_seed")
    projector_kwargs.pop("proj_dim")
    projector_kwargs.pop("proj_dim_dist")

    layer_dim = []
    for layer in layer_name:
        if isinstance(layer, GIPLinear):
            layer_dim.append(layer.weight.shape[0] * layer.weight.shape[1])
        elif isinstance(layer, GIPEmbedding):
            layer_dim.append(layer.embedding_dim * layer.num_embeddings)
        elif isinstance(layer, GIPLayerNorm):
            layer_dim.append(layer.normalized_shape[0])
        else:
            raise ValueError(f"Layer {layer} is not supported")

    if mode == "default" and proj_dim_dist == "non-uniform":
        total_dim = sum(layer_dim)
        proj_dim = [int(proj_dim * dim / total_dim) for dim in layer_dim]
    elif mode in ["one_run", "iterate"] or (mode == "default" and proj_dim_dist == "uniform"):
        proj_dim = [proj_dim] * len(layer_dim)

    print(f"proj_dim: {proj_dim}")
    return projector_kwargs, proj_dim, proj_seed

def eigen_stable_inverse(matrix: torch.Tensor, damping: float = 1e-5, eigen_threshold: float = 1e-6) -> torch.Tensor:
    """
    Compute a numerically stable inverse of a matrix using eigendecomposition.

    Args:
        matrix: Input matrix to invert
        damping: Damping factor for numerical stability
        eigen_threshold: Threshold for small eigenvalues

    Returns:
        Stable inverse of the input matrix
    """
    # Add damping to the diagonal
    matrix = matrix + damping * torch.eye(matrix.shape[0], device=matrix.device)

    # Compute eigendecomposition
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    except RuntimeError:
        # If eigendecomposition fails, try with larger damping
        matrix = matrix + 9 * damping * torch.eye(matrix.shape[0], device=matrix.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

    # Filter small eigenvalues
    eigenvalues = torch.clamp(eigenvalues, min=eigen_threshold)

    # Compute inverse using eigendecomposition
    inverse = torch.matmul(
        eigenvectors,
        torch.matmul(
            torch.diag(1.0 / eigenvalues),
            eigenvectors.t()
        )
    )

    return inverse

class GIPIFAttributorKFAC():
    def __init__(
        self,
        model,
        layer_name: Optional[Union[str, List[str]]] = None,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        damping = 1e-4,
        profile: bool = False,
        mode: str = "default",
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
        self.layer_name = find_GIPlayers(model) if layer_name is None else layer_name
        self.projector_kwargs, self.proj_dim, self.proj_seed = setup_projectors(projector_kwargs, self.layer_name, mode)
        self.damping = damping
        self.profile = profile
        self.mode = mode
        self.device = device
        self.full_train_dataloader = None

        # Initialize profiling stats
        if self.profile:
            self.profiling_stats = {
                'projection': 0.0,
                'gradient': 0.0,
                'hessian': 0.0,
                'inverse_hessian': 0.0
            }

    def KFAC_iHVP(
        self,
        train_dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute K-FAC inverse Hessian vector product for each layer.
        Uses GGN approximation for the Hessian computation.

        Args:
            train_dataloader: DataLoader for training data
            cached_grads: Whether gradients are cached

        Returns:
            Tuple of lists containing the K-FAC factors for each layer
        """
        num_layers = len(self.layer_name)

        Hessian = [0] * num_layers

        train_grad = [None] * len(self.layer_name)
        ihvp_train = [None] * len(self.layer_name)

        num_samples = len(train_dataloader.sampler)

        # Compute K-FAC factors for each layer
        for train_batch_idx, train_batch in enumerate(
            tqdm(
                train_dataloader,
                desc="computing K-FAC factors for training set...",
                leave=False,
            ),
        ):
            train_input_ids = train_batch["input_ids"].to(self.device)
            train_attention_masks = train_batch["attention_mask"].to(self.device)
            train_labels = train_batch["labels"].to(self.device)

            # Time backward pass
            if self.profile:
                torch.cuda.synchronize()
                start_time = time.time()

            # Forward pass
            outputs = self.model(
                input_ids=train_input_ids,
                attention_mask=train_attention_masks,
                labels=train_labels
            )
            logp = -outputs.loss
            train_loss = -(logp - torch.log(1 - torch.exp(logp)))

            train_pre_acts = [layer.pre_activation for layer in self.layer_name]
            Z_grad_train = torch.autograd.grad(train_loss, train_pre_acts, retain_graph=True)

            if self.profile:
                torch.cuda.synchronize()
                self.profiling_stats['gradient'] += time.time() - start_time

            # Compute K-FAC factors for each layer
            with torch.no_grad():
                for layer_id, (layer, z_grad_full) in enumerate(zip(self.layer_name, Z_grad_train)):
                    val_1, val_2 = layer.GIP_components(z_grad_full, per_sample=True)

                    # Apply projection if needed
                    if self.projector_kwargs is not None:
                        # Time projection
                        if self.profile:
                            torch.cuda.synchronize()
                            start_time = time.time()

                        val_1_flatten = val_1.view(-1, val_1.shape[-1])
                        val_2_flatten = val_2.view(-1, val_2.shape[-1])

                        base_seed = self.proj_seed + int(1e4) * layer_id
                        proj_dim = self.proj_dim[layer_id]

                        # input projector
                        random_project_1 = random_project(
                            val_1_flatten,
                            val_1_flatten.shape[0],
                            proj_seed=base_seed,
                            proj_dim=proj_dim,
                            **self.projector_kwargs,
                        )
                        # output_grad projector
                        random_project_2 = random_project(
                            val_2_flatten,
                            val_2_flatten.shape[0],
                            proj_seed=base_seed + 1,
                            proj_dim=proj_dim,
                            **self.projector_kwargs,
                        )

                        # when input is sequence
                        if val_1.dim() == 3:
                            val_1 = random_project_1(val_1_flatten).view(val_1.shape[0], val_1.shape[1], -1)
                            val_2 = random_project_2(val_2_flatten).view(val_2.shape[0], val_2.shape[1], -1)
                        else:
                            val_1 = random_project_1(val_1_flatten)
                            val_2 = random_project_2(val_2_flatten)

                        if self.profile:
                            torch.cuda.synchronize()
                            self.profiling_stats['projection'] += time.time() - start_time

                    batch_size = val_1.shape[0]
                    grad = torch.einsum('BSA,BSC->BAC', val_1, val_2).reshape(batch_size, -1)

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
                        self.profiling_stats['hessian'] += time.time() - start_time


        # Time inverse Hessian calculation
        if self.profile:
            torch.cuda.synchronize()
            start_time = time.time()

        # Add damping term and compute inverses
        for layer_id in range(num_layers):
            # Compute inverses using Cholesky decomposition for stability
            try:
                ihvp_train[layer_id] = torch.matmul(
                    eigen_stable_inverse(Hessian[layer_id], damping=self.damping),
                    train_grad[layer_id].t()
                ).t()
            except RuntimeError as e:
                print(f"Warning: Layer {layer_id} required increased damping for stability")
                ihvp_train[layer_id] = torch.matmul(
                    eigen_stable_inverse(Hessian[layer_id], damping=self.damping * 100),
                    train_grad[layer_id].t()
                ).t()

        if self.profile:
            torch.cuda.synchronize()
            self.profiling_stats['inverse_hessian'] += time.time() - start_time

        return ihvp_train

    def cache(
        self,
        full_train_dataloader: torch.utils.data.DataLoader
    ) -> None:
        # This means we can afford full calculation.
        self.full_train_dataloader = full_train_dataloader
        self.cached_ihvp_train = self.KFAC_iHVP(full_train_dataloader)

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

        if self.mode == "default":
            return self.attribute_default(test_dataloader, train_dataloader)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def attribute_default(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> torch.Tensor:
        """
        Compute influence scores using K-FAC approximation.

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

        # Compute K-FAC factors if not cached
        if train_dataloader is not None and self.full_train_dataloader is None:
            ihvp_train = self.KFAC_iHVP(train_dataloader)
        else:
            ihvp_train = self.cached_ihvp_train

        # Compute influence scores
        for test_batch_idx, test_batch in enumerate(
            tqdm(test_dataloader, desc="computing influence scores...", leave=False),
        ):
            test_input_ids = test_batch["input_ids"].to(self.device)
            test_attention_masks = test_batch["attention_mask"].to(self.device)
            test_labels = test_batch["labels"].to(self.device)

             # Time backward pass
            if self.profile:
                torch.cuda.synchronize()
                start_time = time.time()

            outputs = self.model(
                input_ids=test_input_ids,
                attention_mask=test_attention_masks,
                labels=test_labels
            )
            logp = -outputs.loss
            test_loss = -(logp - torch.log(1 - torch.exp(logp)))

            test_pre_acts = [layer.pre_activation for layer in self.layer_name]
            Z_grad_test = torch.autograd.grad(test_loss, test_pre_acts, retain_graph=True)

            if self.profile:
                torch.cuda.synchronize()
                self.profiling_stats['gradient'] += time.time() - start_time

            with torch.no_grad():
                for layer_id, (layer, z_grad_test) in enumerate(zip(self.layer_name, Z_grad_test)):
                    val_1, val_2 = layer.GIP_components(z_grad_test, per_sample=True)

                    if self.projector_kwargs is not None:
                        if self.profile:
                            torch.cuda.synchronize()
                            start_time = time.time()

                        val_1_flatten = val_1.view(-1, val_1.shape[-1])
                        val_2_flatten = val_2.view(-1, val_2.shape[-1])

                        base_seed = self.proj_seed + int(1e4) * layer_id
                        proj_dim = self.proj_dim[layer_id]

                        # time projection
                        torch.cuda.synchronize()
                        start = time.time()

                        # input projector
                        random_project_1 = random_project(
                            val_1_flatten,
                            val_1_flatten.shape[0],
                            proj_seed=base_seed,
                            proj_dim=proj_dim,
                            **self.projector_kwargs,
                        )
                        # output_grad projector
                        random_project_2 = random_project(
                            val_2_flatten,
                            val_2_flatten.shape[0],
                            proj_seed=base_seed + 1,
                            proj_dim=proj_dim,
                            **self.projector_kwargs,
                        )

                        # when input is sequence
                        if val_1.dim() == 3:
                            val_1 = random_project_1(val_1_flatten).view(val_1.shape[0], val_1.shape[1], -1)
                            val_2 = random_project_2(val_2_flatten).view(val_2.shape[0], val_2.shape[1], -1)
                        else:
                            val_1 = random_project_1(val_1_flatten)
                            val_2 = random_project_2(val_2_flatten)

                        if self.profile:
                            torch.cuda.synchronize()
                            self.profiling_stats['projection'] += time.time() - start_time

                    batch_size = val_1.shape[0]
                    test_grad = torch.einsum('BSA,BSC->BAC', val_1, val_2).reshape(batch_size, -1)

                    col_st = test_batch_idx * test_dataloader.batch_size
                    col_ed = min(
                        (test_batch_idx + 1) * test_dataloader.batch_size,
                        len(test_dataloader.sampler),
                    )

                    # Use grad_dotprod between pre-computed iHVP components and test gradients
                    result = torch.matmul(ihvp_train[layer_id], test_grad.t())

                    IF_score[:, col_st:col_ed] += result

        return (IF_score, self.profiling_stats) if self.profile else IF_score

    def sparsity(
        self,
        dataloader: torch.utils.data.DataLoader,
        thresholds: List[float] = [0.1, 0.5, 0.9],
    ):
        if dataloader is None:
            raise ValueError("Please provide a dataloader to calculate sparsity.")

        # Initialize dictionary to store sparsity values for each layer and threshold
        sparsity = {threshold: {f"layer_{i}": {"val_1": [], "val_2": []}
                    for i in range(len(self.layer_name))}
                    for threshold in thresholds}

        for batch in tqdm(
            dataloader,
            desc="calculating sparsity of the dataset...",
            leave=False,
        ):
            input_ids = batch["input_ids"].to(self.device)
            attention_masks = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                labels=labels
            )
            logp = -outputs.loss
            loss = -(logp - torch.log(1 - torch.exp(logp)))


            pre_acts = [layer.pre_activation for layer in self.layer_name]
            Z_grad = torch.autograd.grad(loss, pre_acts, retain_graph=True)

            with torch.no_grad():
                for layer_id, (layer, z_grad_full) in enumerate(zip(self.layer_name, Z_grad)):
                    val_1, val_2 = layer.GIP_components(z_grad_full, per_sample=True)

                    # Calculate sparsity for each threshold
                    for threshold in thresholds:
                        # Calculate ratio of elements below threshold
                        val_1_sparsity = (torch.abs(val_1) < threshold).float().mean().item()
                        val_2_sparsity = (torch.abs(val_2) < threshold).float().mean().item()

                        # Store results
                        sparsity[threshold][f"layer_{layer_id}"]["val_1"].append(val_1_sparsity)
                        sparsity[threshold][f"layer_{layer_id}"]["val_2"].append(val_2_sparsity)

        # Average sparsity across batches
        for threshold in thresholds:
            for layer_id in range(len(self.layer_name)):
                for val_type in ["val_1", "val_2"]:
                    sparsity[threshold][f"layer_{layer_id}"][val_type] = \
                        sum(sparsity[threshold][f"layer_{layer_id}"][val_type]) / len(sparsity[threshold][f"layer_{layer_id}"][val_type])

        return sparsity