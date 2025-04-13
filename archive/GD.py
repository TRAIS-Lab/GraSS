from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import List, Optional, Union, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

import time

class GradDotAttributor():
    def __init__(
        self,
        model,
        layer_name: Optional[Union[str, List[str]]],
        lr: float = 1e-3,
        profile: bool = False,
        device: str = 'cpu'
    ) -> None:
        """Ghost Inner Product Attributor for Gradient Dot.

        Args:
            model (_type_): _description_
            layer_name (Optional[Union[str, List[str]]], optional): _description_. Defaults to None.
            lr (float, optional): _description_. Defaults to 1e-3.
            profile (bool, optional): Record time used in various parts of the algorithm run. Defaults to False.
            device (str, optional): _description_. Defaults to 'cpu'.
        """
        self.model = model
        self.layer_name = layer_name
        self.lr = lr
        self.profile = profile #TODO implement this
        self.device = device
        self.full_train_dataloader = None

    def cache(
        self,
        full_train_dataloader: torch.utils.data.DataLoader
    ) -> None:
        # This means we can afford full calculation.
        self.full_train_dataloader = full_train_dataloader
        self.cached_train_grad_comp_1 = [None] * len(self.layer_name)
        self.cached_train_grad_comp_2 = [None] * len(self.layer_name)

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
                    grad_comp_1, grad_comp_2 = layer.grad_comp(z_grad_full, per_sample=True)

                    if self.cached_train_grad_comp_1[layer_id] is None:
                        total_samples = len(self.full_train_dataloader.sampler)
                        self.cached_train_grad_comp_1[layer_id] = torch.zeros((total_samples, *grad_comp_1.shape[1:]), device=self.device)
                        self.cached_train_grad_comp_2[layer_id] = torch.zeros((total_samples, *grad_comp_2.shape[1:]), device=self.device)

                    col_st = train_batch_idx * self.full_train_dataloader.batch_size
                    col_ed = min(
                        (train_batch_idx + 1) * self.full_train_dataloader.batch_size,
                        len(self.full_train_dataloader.sampler),
                    )
                    self.cached_train_grad_comp_1[layer_id][col_st:col_ed] = grad_comp_1.detach()
                    self.cached_train_grad_comp_2[layer_id][col_st:col_ed] = grad_comp_2.detach()

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

        tda_output = self.attribute_default(test_dataloader, train_dataloader)

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
            train_grad_comp_1 = [None] * len(self.layer_name)
            train_grad_comp_2 = [None] * len(self.layer_name)

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
                torch.cuda.synchronize(self.device)
                start = time.time()

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

                torch.cuda.synchronize(self.device)
                end = time.time()
                time_backward += end - start

                with torch.no_grad():
                    for layer_id, (layer, z_grad_full) in enumerate(zip(self.layer_name, Z_grad_train)):
                        grad_comp_1, grad_comp_2 = layer.grad_comp(z_grad_full, per_sample=True)

                        if train_grad_comp_1[layer_id] is None:
                            total_samples = len(train_dataloader.sampler)
                            train_grad_comp_1[layer_id] = torch.zeros((total_samples, *grad_comp_1.shape[1:]), device=self.device)
                            train_grad_comp_2[layer_id] = torch.zeros((total_samples, *grad_comp_2.shape[1:]), device=self.device)

                        col_st = train_batch_idx * train_dataloader.batch_size
                        col_ed = min(
                            (train_batch_idx + 1) * train_dataloader.batch_size,
                            len(train_dataloader.sampler),
                        )
                        train_grad_comp_1[layer_id][col_st:col_ed] = grad_comp_1.detach()
                        train_grad_comp_2[layer_id][col_st:col_ed] = grad_comp_2.detach()
        else:
            train_grad_comp_1, train_grad_comp_2 = self.cached_train_grad_comp_1, self.cached_train_grad_comp_2

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
            test_loss = logp - torch.log(1 - torch.exp(logp))

            # Get pre-activations from trainable layers
            test_pre_acts = [layer.pre_activation for layer in self.layer_name]

            # time backward pass
            torch.cuda.synchronize(self.device)
            start = time.time()

            # Calculate gradients
            Z_grad_test = torch.autograd.grad(test_loss, test_pre_acts, retain_graph=True)

            torch.cuda.synchronize(self.device)
            end = time.time()
            time_backward += end - start

            # Calculate scores
            with torch.no_grad():
                for layer_id, (layer, z_grad_test) in enumerate(zip(self.layer_name, Z_grad_test)):
                    grad_comp_1, grad_comp_2 = layer.grad_comp(z_grad_test, per_sample=True)

                    # time inner product
                    torch.cuda.synchronize(self.device)
                    start = time.time()

                    result = layer.grad_dot_prod_from_grad_comp(train_grad_comp_1[layer_id], train_grad_comp_2[layer_id], grad_comp_1, grad_comp_2)

                    torch.cuda.synchronize(self.device)
                    end = time.time()
                    time_inner_product += end - start

                    col_st = test_batch_idx * test_dataloader.batch_size
                    col_ed = min(
                        (test_batch_idx + 1) * test_dataloader.batch_size,
                        len(test_dataloader.sampler),
                    )
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
        """deprecate method."""
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
        full_loss = logp - torch.log(1 - torch.exp(logp))

        # Get pre-activations from trainable layers
        full_pre_acts = [layer.pre_activation for layer in self.layer_name]

        # Calculate gradients
        Z_grad_full = torch.autograd.grad(full_loss, full_pre_acts, retain_graph=True)

        # Calculate scores
        with torch.no_grad():
            for layer_id, (layer, z_grad_full) in enumerate(zip(self.layer_name, Z_grad_full)):
                grad_comp_1, grad_comp_2 = layer.grad_comp(z_grad_full, per_sample=True)

                result = layer.grad_dotprod(grad_comp_1[:num_train], grad_comp_2[:num_train], grad_comp_1[num_train:], grad_comp_2[num_train:])

                grad_dot += result * self.lr

        return grad_dot

    def attribute_iterate(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: torch.utils.data.DataLoader,
    ) -> Tensor:
        """deprecate method."""
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
