from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import List, Optional, Union

import torch
from torch import Tensor
from tqdm import tqdm

from .layers.linear import GIPLinear, GIPEmbedding
from .layers.layer_norm import GIPLayerNorm

from _dattri.func.projection import random_project

def find_GIPlayers(model):
    GIP_layers = []

    for module in model.modules():
        if isinstance(module, GIPLinear) or isinstance(module, GIPLayerNorm) or isinstance(module, GIPEmbedding):
            GIP_layers.append(module)

    return GIP_layers

def grad_dotprod(A1, B1, A2, B2) -> Tensor:
    """Compute gradient sample norm for the weight matrix in a GIPlinear layer."""
    if A1.dim() == 2 and B1.dim() == 2:
        dot_prod_1 = torch.matmul(A1, A2.T)
        dot_prod_2 = torch.matmul(B1, B2.T)
        dot_prod = dot_prod_1*dot_prod_2

        return dot_prod
    elif A1.dim() == 3 and B1.dim() == 3:
        (b, t, p), (_, _, d) = A1.size(), B1.size()
        nval, _, _ = A2.size()

        if 2*b*nval*t**2 < (b+nval)*p*d:

            #transpose_start = time.time()
            A2, B2 = A2.transpose(-1, -2), B2.transpose(-1, -2)
            #transpose_end = time.time()
            #print('Time for transpose: {}'.format(transpose_end - transpose_start))

            A1_expanded = A1.unsqueeze(1)
            A2_expanded = A2.unsqueeze(0)
            B1_expanded = B1.unsqueeze(1)
            B2_expanded = B2.unsqueeze(0)

            # expand_end = time.time()
            #print('Time for expand: {}'.format(expand_end - transpose_end))

            # Memory consumption: 2*b*nval*T^2
            # A_dotprod = torch.matmul(A1_expanded, A2_expanded) # Shape: [b, nval, T, T]
            # B_dotprod = torch.matmul(B1_expanded, B2_expanded) # Shape: [b, nval, T, T]
            A_dotprod = _chunked_matmul(A1_expanded, A2_expanded, chunk_size=4096)
            B_dotprod = _chunked_matmul(B1_expanded, B2_expanded, chunk_size=4096)

            # chunk_end = time.time()
            # print('Time for chunked matmul: {}'.format(chunk_end - expand_end))

            result = (A_dotprod * B_dotprod).sum(dim=(2, 3))
            #result_end = time.time()
            #print('Time for sum: {}'.format(result_end - chunk_end))

            return result

        else:

            # [b, p, T] * [b, T, d]
            A = torch.bmm(B1.permute(0, 2, 1), A1).flatten(start_dim=1) # Shape: [b, p*d]
            B = torch.bmm(B2.permute(0, 2, 1), A2).flatten(start_dim=1) # Shape: [nval, p*d]

            return torch.matmul(A, B.T)
    else:
        raise ValueError(f"Unexpected input shape: {A1.size()}, grad_output shape: {B1.size()}")

def _chunked_matmul(A1, A2, chunk_size=128):
    """
    Performs matrix multiplication in chunks for memory efficiency.

    Parameters:
    A1 (torch.Tensor): The first tensor with shape [n1, c1, h1, w1]
    A2 (torch.Tensor): The second tensor with shape [n2, c2, w2, h2]
    chunk_size (int): The size of each chunk to be multiplied

    Returns:
    torch.Tensor: The result of the matrix multiplication with shape [n1, c2, h1, h2]
    """
    # Validate input shapes
    if A1.shape[-1] != A2.shape[-2]:
        raise ValueError(f"Inner dimensions must match for matrix multiplication, got {A1.shape[-1]} and {A2.shape[-2]}")

    # Determine output shape
    n1, c1, h1, w1 = A1.shape
    n2, c2, w2, h2 = A2.shape

    if w1 != w2:
        raise ValueError(f"Inner matrix dimensions must agree, got {w1} and {w2}")

    # Prepare the result tensor on the same device as the inputs
    result = torch.zeros(n1, c2, h1, h2, device=A1.device, dtype=A1.dtype)

    # Perform the multiplication in chunks
    for start in range(0, w1, chunk_size):
        end = min(start + chunk_size, w1)
        A1_chunk = A1[:, :, :, start:end]  # [8, 1, 1024, chunk_size]
        A2_chunk = A2[:, :, start:end, :]  # [1, 8, chunk_size, 1024]

        # Multiply the chunks
        result += torch.matmul(A1_chunk, A2_chunk)

    return result

class GhostInnerProductAttributor():
    def __init__(
        self,
        model,
        lr: float = 1e-3,
        layer_name: Optional[Union[str, List[str]]] = None,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        mode: str = "default",
        device: str = 'cpu'
    ) -> None:

        self.model = model
        self.lr = lr
        self.layer_name = find_GIPlayers(model) if layer_name is None else layer_name

        # need more complex control of proj_seed and proj_dim for different layers, extract first
        if projector_kwargs is not None:
            self.proj_seed = projector_kwargs.get("proj_seed", 0)
            proj_dim = projector_kwargs.get("proj_dim", 512)

            projector_kwargs.pop("proj_seed")
            projector_kwargs.pop("proj_dim")

            layer_dim = []
            for layer in self.layer_name:
                if isinstance(layer, GIPLinear):
                    layer_dim.append(layer.weight.shape[0] * layer.weight.shape[1])
                elif isinstance(layer, GIPEmbedding):
                    layer_dim.append(layer.embedding_dim * layer.num_embeddings)
                elif isinstance(layer, GIPLayerNorm):
                    layer_dim.append(layer.normalized_shape[0])
                else:
                    raise ValueError(f"Layer {layer} is not supported for Ghost Inner Product.")

            if mode == "default": # for default mode, we will store all the pre-activation gradient and input of the training set, hence the total projection dimension need to be proj_dim
                total_dim = sum(layer_dim)
                # distribute proj_dim in proj_kwargs to each layer w.r.t. their parameter count
                self.proj_dim = [int(proj_dim * dim / total_dim) for dim in layer_dim]
            elif mode in ["one_run", "iterate"]: # for one_run and iterate mode, we will process layer by layer, so we can set the proj_dim to be the same for all layers (we never materialize the full pre-activation gradient and input of the training set, so this is affordable)
                self.proj_dim = [proj_dim] * len(self.layer_name)

            print(f"proj_dim: {self.proj_dim}")

        self.projector_kwargs = projector_kwargs
        self.mode = mode
        self.device = device

        self.full_train_dataloader = None

    def cache(
        self,
        full_train_dataloader: torch.utils.data.DataLoader
    ) -> None:
        # This means we can afford full calculation.
        self.full_train_dataloader = full_train_dataloader
        self.cached_train_val_1 = [[] for _ in range(len(self.layer_name))]
        self.cached_train_val_2 = [[] for _ in range(len(self.layer_name))]

        for train_batch in tqdm(
            self.full_train_dataloader,
            desc="calculating gradient of training set...",
            leave=False,
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
            train_loss = outputs.loss

            # Get pre-activations from trainable layers
            train_pre_acts = [layer.pre_activation for layer in self.layer_name]

            # Calculate gradients
            Z_grad_train = torch.autograd.grad(train_loss, train_pre_acts, retain_graph=True)

            with torch.no_grad():
                for layer_id, (layer, z_grad_full) in enumerate(zip(self.layer_name, Z_grad_train)):
                    val_1, val_2 = layer.pe_grad_gradcomp(z_grad_full, per_sample=True)
                    if self.projector_kwargs is not None:
                        val_1_flatten = val_1.view(-1, val_1.shape[1])
                        val_2_flatten = val_2.view(-1, val_1.shape[1])

                        # input projector
                        random_project_1 = random_project(
                            val_1_flatten,
                            val_1_flatten.shape[0],
                            proj_seed=self.proj_seed + int(1e4) * layer_id,
                            proj_dim=self.proj_dim[layer_id],
                            **self.projector_kwargs,
                        )
                        # output_grad projector
                        random_project_2 = random_project(
                            val_2_flatten,
                            val_2_flatten.shape[0],
                            proj_seed=self.proj_seed + int(1e4) * layer_id + 1,
                            proj_dim=self.proj_dim[layer_id],
                            **self.projector_kwargs,
                        )

                        # when input is sequence
                        if val_1.dim() == 3:
                            val_1 = random_project_1(val_1_flatten).view(val_1.shape[0], -1, val_1.shape[2])
                            val_2 = random_project_2(val_2_flatten).view(val_2.shape[0], -1, val_2.shape[2])
                        else:
                            val_1 = random_project_1(val_1_flatten)
                            val_2 = random_project_2(val_2_flatten)

                    # Append to the appropriate layer's list
                    self.cached_train_val_1[layer_id].append(val_1.detach())
                    self.cached_train_val_2[layer_id].append(val_2.detach())

        for layer_id in range(len(self.layer_name)):
            self.cached_train_val_1[layer_id] = torch.cat(self.cached_train_val_1[layer_id], dim=0)
            self.cached_train_val_2[layer_id] = torch.cat(self.cached_train_val_2[layer_id], dim=0)

    def attribute(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Tensor:
        """Attributing the test set with respect to the training set.

        Args:
            test_dataloader (torch.utils.data.DataLoader): _description_
            train_dataloader (Optional[torch.utils.data.DataLoader], optional): _description_. Defaults to None.
            mode (str, optional): There are several mode of ghost inner product:

                1. "default": first compute (if not cached) and store all the pre-activation gradient and input of the training set, then for the test set. Do the inner product at the end. This is much like the vanilla Grad-Dot, only differ in that we derive the gradient inner product from a different formula, so we require different terms.

                The memory requirement is the same as vanilla Grad-Dot.

                2. "one_run": Original Ghost Inner Product. Compute the gradient inner product of the training set and the test set in one run.

                This requires a lot of memory to store be able to compute the gradient of the pre-activation and the input of the training+test set.

                3. "iterate": Iterate through training batches and test batches and use "one_run" for each pair of batches.

                This is the most memory efficient way to compute the inner product, but it is also the slowest due to the repetition of the computation.

            Defaults to "default".

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

        if train_dataloader is not None: # not cached
            train_val_1 = [[] for _ in range(len(self.layer_name))]
            train_val_2 = [[] for _ in range(len(self.layer_name))]

            for train_batch in tqdm(
                train_dataloader,
                desc="calculating gradient of training set...",
                leave=False,
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
                train_loss = outputs.loss

                # Get pre-activations from trainable layers
                train_pre_acts = [layer.pre_activation for layer in self.layer_name]

                # Calculate gradients
                Z_grad_train = torch.autograd.grad(train_loss, train_pre_acts, retain_graph=True)

                with torch.no_grad():
                    for layer_id, (layer, z_grad_full) in enumerate(zip(self.layer_name, Z_grad_train)):
                        val_1, val_2 = layer.pe_grad_gradcomp(z_grad_full, per_sample=True)
                        if self.projector_kwargs is not None:
                            val_1_flatten = val_1.view(-1, val_1.shape[1])
                            val_2_flatten = val_2.view(-1, val_1.shape[1])

                            # input projector
                            random_project_1 = random_project(
                                val_1_flatten,
                                val_1_flatten.shape[0],
                                proj_seed=self.proj_seed + int(1e4) * layer_id,
                                proj_dim=self.proj_dim[layer_id],
                                **self.projector_kwargs,
                            )
                            # output_grad projector
                            random_project_2 = random_project(
                                val_2_flatten,
                                val_2_flatten.shape[0],
                                proj_seed=self.proj_seed + int(1e4) * layer_id + 1,
                                proj_dim=self.proj_dim[layer_id],
                                **self.projector_kwargs,
                            )

                            # when input is sequence
                            if val_1.dim() == 3:
                                val_1 = random_project_1(val_1_flatten).view(val_1.shape[0], -1, val_1.shape[2])
                                val_2 = random_project_2(val_2_flatten).view(val_2.shape[0], -1, val_2.shape[2])
                            else:
                                val_1 = random_project_1(val_1_flatten)
                                val_2 = random_project_2(val_2_flatten)

                        train_val_1[layer_id].append(val_1.detach())
                        train_val_2[layer_id].append(val_2.detach())

            for layer_id in range(len(self.layer_name)):
                train_val_1[layer_id] = torch.cat(train_val_1[layer_id], dim=0)
                train_val_2[layer_id] = torch.cat(train_val_2[layer_id], dim=0)
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

            outputs = self.model(
                input_ids=test_input_ids,
                attention_mask=test_attention_masks,
                labels=test_labels
            )
            test_loss = outputs.loss

            # Forward pass with all data
            outputs = self.model(
                input_ids=test_input_ids,
                attention_mask=test_attention_masks,
                labels=test_labels
            )
            test_loss = outputs.loss

            # Get pre-activations from trainable layers
            test_pre_acts = [layer.pre_activation for layer in self.layer_name]

            # Calculate gradients
            Z_grad_test = torch.autograd.grad(test_loss, test_pre_acts, retain_graph=True)

            # Calculate scores
            with torch.no_grad():
                for layer_id, (layer, z_grad_test) in enumerate(zip(self.layer_name, Z_grad_test)):
                    val_1, val_2 = layer.pe_grad_gradcomp(z_grad_test, per_sample=True)
                    if self.projector_kwargs is not None:
                        val_1_flatten = val_1.view(-1, val_1.shape[1])
                        val_2_flatten = val_2.view(-1, val_1.shape[1])

                        # input projector
                        random_project_1 = random_project(
                            val_1_flatten,
                            val_1_flatten.shape[0],
                            proj_seed=self.proj_seed + int(1e4) * layer_id,
                            proj_dim=self.proj_dim[layer_id],
                            **self.projector_kwargs,
                        )
                        # output_grad projector
                        random_project_2 = random_project(
                            val_2_flatten,
                            val_2_flatten.shape[0],
                            proj_seed=self.proj_seed + int(1e4) * layer_id + 1,
                            proj_dim=self.proj_dim[layer_id],
                            **self.projector_kwargs,
                        )

                        # when input is sequence
                        if val_1.dim() == 3:
                            val_1 = random_project_1(val_1_flatten).view(val_1.shape[0], -1, val_1.shape[2])
                            val_2 = random_project_2(val_2_flatten).view(val_2.shape[0], -1, val_2.shape[2])
                        else:
                            val_1 = random_project_1(val_1_flatten)
                            val_2 = random_project_2(val_2_flatten)

                    dLdZ_train, z_train = train_val_1[layer_id], train_val_2[layer_id]
                    dLdZ_val, z_val = val_1, val_2
                    col_st = test_batch_idx * test_dataloader.batch_size
                    col_ed = min(
                        (test_batch_idx + 1) * test_dataloader.batch_size,
                        len(test_dataloader.sampler),
                    )
                    result = grad_dotprod(dLdZ_train, z_train, dLdZ_val, z_val) * self.lr
                    grad_dot[:, col_st:col_ed] += result

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
        print(train_input_ids)
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
        full_loss = outputs.loss

        # Get pre-activations from trainable layers
        full_pre_acts = [layer.pre_activation for layer in self.layer_name]

        # Calculate gradients
        Z_grad_full = torch.autograd.grad(full_loss, full_pre_acts, retain_graph=True)

        # Calculate scores
        with torch.no_grad():
            for layer_id, (layer, z_grad_full) in enumerate(zip(self.layer_name, Z_grad_full)):
                val_1, val_2 = layer.pe_grad_gradcomp(z_grad_full, per_sample=True)
                if self.projector_kwargs is not None:
                    val_1_flatten = val_1.view(-1, val_1.shape[1])
                    val_2_flatten = val_2.view(-1, val_1.shape[1])

                    # input projector
                    random_project_1 = random_project(
                        val_1_flatten,
                        val_1_flatten.shape[0],
                        proj_seed=self.proj_seed + int(1e4) * layer_id,
                        proj_dim=self.proj_dim[layer_id],
                        **self.projector_kwargs,
                    )
                    # output_grad projector
                    random_project_2 = random_project(
                        val_2_flatten,
                        val_2_flatten.shape[0],
                        proj_seed=self.proj_seed + int(1e4) * layer_id + 1,
                        proj_dim=self.proj_dim[layer_id],
                        **self.projector_kwargs,
                    )

                    # when input is sequence
                    if val_1.dim() == 3:
                        val_1 = random_project_1(val_1_flatten).view(val_1.shape[0], -1, val_1.shape[2])
                        val_2 = random_project_2(val_2_flatten).view(val_2.shape[0], -1, val_2.shape[2])
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