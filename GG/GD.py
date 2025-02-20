from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import List, Optional, Union, Tuple

import torch
from torch import Tensor
from tqdm import tqdm
from .layers.linear import GGLinear, GGEmbedding
from .layers.layer_norm import GGLayerNorm
from .helper import find_GGlayers, grad_dotprod

from _dattri.func.projection import random_project

import time

def setup_projectors(
        projector_kwargs: Dict[str, Any],
        layer_name: List[Union[GGLinear, GGEmbedding, GGLayerNorm]],
        mode: Optional[str] = "default",
    ) -> Tuple[Dict[str, Any], List[int], int]:
    """Setup projection dimensions and seeds for each layer.

    Args:
        projector_kwargs (Dict[str, Any]): projector's arguments.
        layer_name (List[Union[GGLinear, GGEmbedding, GGLayerNorm]]): the list of layers to be projected.
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
        if isinstance(layer, GGLinear):
            layer_dim.append(layer.weight.shape[0] * layer.weight.shape[1])
        elif isinstance(layer, GGEmbedding):
            layer_dim.append(layer.embedding_dim * layer.num_embeddings)
        elif isinstance(layer, GGLayerNorm):
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

class GGGradDotAttributor():
    def __init__(
        self,
        model,
        lr: float = 1e-3,
        layer_name: Optional[Union[str, List[str]]] = None,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        mode: str = "default",
        device: str = 'cpu'
    ) -> None:
        """Ghost Inner Product Attributor for Gradient Dot.

        Args:
            model (_type_): _description_
            lr (float, optional): _description_. Defaults to 1e-3.
            layer_name (Optional[Union[str, List[str]]], optional): _description_. Defaults to None.
            projector_kwargs (Optional[Dict[str, Any]], optional): _description_. Defaults to None.
            mode (str, optional): There are several mode of ghost inner product:

                1. "default": first compute (if not cached) and store all the pre-activation gradient and input of the training set, then for the test set. Do the inner product at the end. This is much like the vanilla Grad-Dot, only differ in that we derive the gradient inner product from a different formula, so we require different terms.

                The memory requirement is the same as vanilla Grad-Dot.

                2. "one_run": Original Ghost Inner Product. Compute the gradient inner product of the training set and the test set in one run.

                This requires a lot of memory to store be able to compute the gradient of the pre-activation and the input of the training+test set.

                3. "iterate": Iterate through training batches and test batches and use "one_run" for each pair of batches.

                This is the most memory efficient way to compute the inner product, but it is also the slowest due to the repetition of the computation.

             Defaults to "default".
            device (str, optional): _description_. Defaults to 'cpu'.
        """
        self.model = model
        self.lr = lr
        self.layer_name = find_GGlayers(model) if layer_name is None else layer_name
        self.projector_kwargs, self.proj_dim, self.proj_seed = setup_projectors(projector_kwargs, self.layer_name, mode)
        self.mode = mode
        self.device = device
        self.full_train_dataloader = None

    def cache(
        self,
        full_train_dataloader: torch.utils.data.DataLoader
    ) -> None:
        # This means we can afford full calculation.
        self.full_train_dataloader = full_train_dataloader
        self.cached_train_val_1 = [None] * len(self.layer_name)
        self.cached_train_val_2 = [None] * len(self.layer_name)

        for train_batch_idx, train_batch in enumerate(
            tqdm(
                self.full_train_dataloader,
                desc="calculating gradient of training set...",
                leave=False,
            ),
        ):
            train_input_ids = train_batch["input_ids"].to(self.device)
            train_attention_masks = train_batch["attention_mask"].to(self.device)
            train_labels = train_batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=train_input_ids,
                attention_mask=train_attention_masks,
                labels=train_labels
            )
            logp = -outputs.loss
            train_loss = logp - torch.log(1 - torch.exp(logp))

            # Get pre-activations from trainable layers
            train_pre_acts = [layer.pre_activation for layer in self.layer_name]

            # Calculate gradients
            Z_grad_train = torch.autograd.grad(train_loss, train_pre_acts, retain_graph=True)

            with torch.no_grad():
                for layer_id, (layer, z_grad_full) in enumerate(zip(self.layer_name, Z_grad_train)):
                    val_1, val_2 = layer.per_example_gradient(z_grad_full, per_sample=True)
                    if self.projector_kwargs is not None:
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

                    if self.cached_train_val_1[layer_id] is None:
                        total_samples = len(self.full_train_dataloader.sampler)
                        self.cached_train_val_1[layer_id] = torch.zeros((total_samples, *val_1.shape[1:]), device=self.device)
                        self.cached_train_val_2[layer_id] = torch.zeros((total_samples, *val_2.shape[1:]), device=self.device)

                    col_st = train_batch_idx * self.full_train_dataloader.batch_size
                    col_ed = min(
                        (train_batch_idx + 1) * self.full_train_dataloader.batch_size,
                        len(self.full_train_dataloader.sampler),
                    )
                    self.cached_train_val_1[layer_id][col_st:col_ed] = val_1.detach()
                    self.cached_train_val_2[layer_id][col_st:col_ed] = val_2.detach()

    def attribute(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
        reverse: bool = False,
    ) -> Tensor:
        """Attributing the test set with respect to the training set.

        Args:
            test_dataloader (torch.utils.data.DataLoader): _description_
            train_dataloader (Optional[torch.utils.data.DataLoader], optional): _description_. Defaults to None.
            reverse (bool, optional): _description_. Defaults to False.

        Returns:
            Tensor: The gradient inner product of the training set and the test set.
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

        if reverse and self.full_train_dataloader is not None:
            message = "You can't reverse the attribution when you have cached the training loader."
            raise ValueError(message)
        elif reverse:
            test_dataloader, train_dataloader = train_dataloader, test_dataloader

        if self.mode == "default":
            tda_output = self.attribute_default(test_dataloader, train_dataloader)
        elif self.mode == "one_run":
            if self.full_train_dataloader is not None:
                message = "One run can't be cached."
                raise ValueError(message)
            tda_output = self.attribute_one_run(test_dataloader, train_dataloader)
        elif self.mode == "iterate":
            if self.full_train_dataloader is not None:
                message = "Iterate can't be cached."
                raise ValueError(message)
            tda_output = self.attribute_iterate(test_dataloader, train_dataloader)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if reverse:
            tda_output = tda_output.T
        return tda_output

    def attribute_default(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Tensor:
        """Might be cached or not cached. If cached, train_dataloader is None. If not cached, train_dataloader is not None.

        Args:
            test_dataloader (torch.utils.data.DataLoader)
            train_dataloader (Optional[torch.utils.data.DataLoader], optional). Defaults to None.

        Returns:
            Tensor.
        """
        if train_dataloader is not None:
            num_train = len(train_dataloader.sampler)
        else:
            num_train = len(self.full_train_dataloader.sampler)

        grad_dot = torch.zeros(num_train, len(test_dataloader.sampler), device=self.device)

        time_projection = 0
        time_inner_product = 0
        time_backward = 0

        if train_dataloader is not None: # not cached
            train_val_1 = [None] * len(self.layer_name)
            train_val_2 = [None] * len(self.layer_name)

            for train_batch_idx, train_batch in enumerate(
                tqdm(
                    train_dataloader,
                    desc="calculating gradient of training set...",
                    leave=False,
                ),
            ):
                train_input_ids = train_batch["input_ids"].to(self.device)
                train_attention_masks = train_batch["attention_mask"].to(self.device)
                train_labels = train_batch["labels"].to(self.device)

                # time backward pass
                torch.cuda.synchronize()
                start = time.time()

                # Forward pass
                outputs = self.model(
                    input_ids=train_input_ids,
                    attention_mask=train_attention_masks,
                    labels=train_labels
                )
                logp = -outputs.loss
                train_loss = logp - torch.log(1 - torch.exp(logp))
                # train_loss = -(logp - torch.log(1 - torch.exp(logp)))

                # Get pre-activations from trainable layers
                train_pre_acts = [layer.pre_activation for layer in self.layer_name]

                # Calculate gradients
                Z_grad_train = torch.autograd.grad(train_loss, train_pre_acts, retain_graph=True)

                torch.cuda.synchronize()
                end = time.time()
                time_backward += end - start

                with torch.no_grad():
                    for layer_id, (layer, z_grad_full) in enumerate(zip(self.layer_name, Z_grad_train)):
                        val_1, val_2 = layer.per_example_gradient(z_grad_full, per_sample=True)
                        if self.projector_kwargs is not None:
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

                            torch.cuda.synchronize()
                            end = time.time()
                            time_projection += end - start

                        if train_val_1[layer_id] is None:
                            total_samples = len(train_dataloader.sampler)
                            train_val_1[layer_id] = torch.zeros((total_samples, *val_1.shape[1:]), device=self.device)
                            train_val_2[layer_id] = torch.zeros((total_samples, *val_2.shape[1:]), device=self.device)

                        col_st = train_batch_idx * train_dataloader.batch_size
                        col_ed = min(
                            (train_batch_idx + 1) * train_dataloader.batch_size,
                            len(train_dataloader.sampler),
                        )
                        train_val_1[layer_id][col_st:col_ed] = val_1.detach()
                        train_val_2[layer_id][col_st:col_ed] = val_2.detach()
        else:
            train_val_1, train_val_2 = self.cached_train_val_1, self.cached_train_val_2

        for test_batch_idx, test_batch in enumerate(
            tqdm(
                test_dataloader,
                desc="calculating gradient of evaluation set...",
                leave=False,
            ),
        ):
            test_input_ids = test_batch["input_ids"].to(self.device)
            test_attention_masks = test_batch["attention_mask"].to(self.device)
            test_labels = test_batch["labels"].to(self.device)

            # Forward pass with all data
            outputs = self.model(
                input_ids=test_input_ids,
                attention_mask=test_attention_masks,
                labels=test_labels
            )
            logp = -outputs.loss

            # test_loss = -(logp - torch.log(1 - torch.exp(logp)))
            test_loss = logp - torch.log(1 - torch.exp(logp))

            # Get pre-activations from trainable layers
            test_pre_acts = [layer.pre_activation for layer in self.layer_name]

            # time backward pass
            torch.cuda.synchronize()
            start = time.time()

            # Calculate gradients
            Z_grad_test = torch.autograd.grad(test_loss, test_pre_acts, retain_graph=True)

            torch.cuda.synchronize()
            end = time.time()
            time_backward += end - start

            # Calculate scores
            with torch.no_grad():
                for layer_id, (layer, z_grad_test) in enumerate(zip(self.layer_name, Z_grad_test)):
                    print(layer)
                    val_1, val_2 = layer.per_example_gradient(z_grad_test, per_sample=True)
                    if self.projector_kwargs is not None:
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

                        torch.cuda.synchronize()
                        end = time.time()
                        time_projection += end - start

                    col_st = test_batch_idx * test_dataloader.batch_size
                    col_ed = min(
                        (test_batch_idx + 1) * test_dataloader.batch_size,
                        len(test_dataloader.sampler),
                    )

                    # time inner product
                    torch.cuda.synchronize()
                    start = time.time()

                    if isinstance(layer, GGLinear):
                        dLdZ_train, z_train = train_val_1[layer_id], train_val_2[layer_id]
                        dLdZ_val, z_val = val_1, val_2

                        grad = torch.einsum('BSA,BSC->BAC', dLdZ_train, z_train).reshape(dLdZ_train.shape[0], -1)

                        print(grad.shape)
                        print(grad[:, :3])

                        result = grad_dotprod(dLdZ_train, z_train, dLdZ_val, z_val)
                    elif isinstance(layer, GGLayerNorm):
                        dLdgamma_train, dLdbeta_train = train_val_1[layer_id], train_val_2[layer_id]
                        dLdgamma_val, dLdbeta_val = val_1, val_2

                        print(dLdgamma_train.shape)
                        print(dLdgamma_train[:, :3])

                        print(dLdbeta_train.shape)
                        print(dLdbeta_train[:, :3])

                        result = (dLdgamma_train @ dLdgamma_val.T + dLdbeta_train @ dLdbeta_val.T)
                    torch.cuda.synchronize()
                    end = time.time()
                    time_inner_product += end - start

                    grad_dot[:, col_st:col_ed] += result * self.lr

        print(f"Time for projection: {time_projection}")
        print(f"Time for inner product: {time_inner_product}")
        print(f"Time for backward pass: {time_backward}")
        return grad_dot

    def attribute_one_run(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: torch.utils.data.DataLoader,
    ) -> Tensor:
        num_train = len(train_dataloader.sampler)
        num_test = len(test_dataloader.sampler)
        grad_dot = torch.zeros(num_train, num_test, device=self.device)

        train_input_ids, train_attention_masks, train_labels = [], [], []
        test_input_ids, test_attention_masks, test_labels = [], [], []

        # Gather training data
        for batch in train_dataloader:
            train_input_ids.append(batch["input_ids"])
            train_attention_masks.append(batch["attention_mask"])
            train_labels.append(batch["labels"])

        # Gather evaluation data
        for batch in test_dataloader:
            test_input_ids.append(batch["input_ids"])
            test_attention_masks.append(batch["attention_mask"])
            test_labels.append(batch["labels"])

        # Concatenate all batches
        train_input_ids = torch.cat(train_input_ids, dim=0).to(self.device)
        train_attention_masks = torch.cat(train_attention_masks, dim=0).to(self.device)
        train_labels = torch.cat(train_labels, dim=0).to(self.device)

        test_input_ids = torch.cat(test_input_ids, dim=0).to(self.device)
        test_attention_masks = torch.cat(test_attention_masks, dim=0).to(self.device)
        test_labels = torch.cat(test_labels, dim=0).to(self.device)

        # Combine all data
        combined_input_ids = torch.cat((train_input_ids, test_input_ids), dim=0)
        combined_attention_masks = torch.cat((train_attention_masks, test_attention_masks), dim=0)
        combined_labels = torch.cat((train_labels, test_labels), dim=0)

        # Forward pass with all data
        outputs = self.model(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_masks,
            labels=combined_labels
        )
        logp = -outputs.loss
        full_loss = -(logp - torch.log(1 - torch.exp(logp)))

        # Get pre-activations from trainable layers
        full_pre_acts = [layer.pre_activation for layer in self.layer_name]

        # Calculate gradients
        Z_grad_full = torch.autograd.grad(full_loss, full_pre_acts, retain_graph=True)

        # Calculate scores
        with torch.no_grad():
            for layer_id, (layer, z_grad_full) in enumerate(zip(self.layer_name, Z_grad_full)):
                val_1, val_2 = layer.per_example_gradient(z_grad_full, per_sample=True)
                if self.projector_kwargs is not None:
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

                dLdZ_train, z_train = val_1[:num_train], val_2[:num_train]
                dLdZ_val, z_val = val_1[num_train:], val_2[num_train:]
                grad_dot += grad_dotprod(dLdZ_train, z_train, dLdZ_val, z_val) * self.lr

        return grad_dot

    def attribute_iterate(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: torch.utils.data.DataLoader,
    ) -> Tensor:
        num_train = len(train_dataloader.sampler)
        num_test = len(test_dataloader.sampler)
        grad_dot = torch.zeros(num_train, num_test, device=self.device)

        # Helper class to wrap a single batch
        class SingleBatchDataset(torch.utils.data.Dataset):
            def __init__(self, batch):
                self.batch = batch

            def __len__(self):
                return self.batch["input_ids"].size(0)

            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.batch.items()}

        for train_batch_idx, train_batch in enumerate(
            tqdm(
                train_dataloader,
                desc="calculating gradient of training set...",
                leave=False,
            ),
        ):
            train_batch_size = train_batch["input_ids"].size(0)
            for test_batch_idx, test_batch in enumerate(
                tqdm(
                    test_dataloader,
                    desc="calculating gradient of evaluation set...",
                    leave=False,
                ),
            ):
                test_batch_size = test_batch["input_ids"].size(0)

                # Create proper datasets from single batches
                train_batch_dataset = SingleBatchDataset(train_batch)
                test_batch_dataset = SingleBatchDataset(test_batch)

                # Create proper dataloaders
                train_batch_loader = torch.utils.data.DataLoader(
                    train_batch_dataset,
                    batch_size=train_batch_size,
                    shuffle=False
                )
                test_batch_loader = torch.utils.data.DataLoader(
                    test_batch_dataset,
                    batch_size=test_batch_size,
                    shuffle=False
                )

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

                grad_dot[row_st:row_ed, col_st:col_ed] += self.attribute_one_run(test_batch_loader, train_batch_loader)

        return grad_dot

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
                    val_1, val_2 = layer.per_example_gradient(z_grad_full, per_sample=True)

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