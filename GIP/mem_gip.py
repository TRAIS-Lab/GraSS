import torch
from torch import Tensor
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
import h5py
import math

from .helper import find_GIPlayers, grad_dotprod, setup_projectors

from _dattri.func.projection import random_project

from torch.profiler import profile, record_function, ProfilerActivity

@dataclass
class ChunkInfo:
    start_idx: int
    size: int
    layer_shapes: List[Tuple[int, ...]]

class MemEffGhostInnerProductAttributor():
    def __init__(
        self,
        model,
        lr: float = 1e-3,
        layer_name: Optional[Union[str, List[str]]] = None,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        device: str = 'cpu',
        cache_dir: str = "./gip_cache",
        chunk_size: int = 5,
        min_chunk_size: int = 3,
        max_test_batches: Optional[int] = None,  # Maximum number of test batches to process at once
    ) -> None:
        self.model = model
        self.lr = lr
        self.layer_name = find_GIPlayers(model) if layer_name is None else layer_name
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        assert chunk_size >= min_chunk_size, "Chunk size must be greater than or equal to min chunk size."
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_test_batches = max_test_batches
        self.chunks: List[ChunkInfo] = []

        # Handle projection setup
        if projector_kwargs is not None:
            self.projector_kwargs, self.proj_dim, self.proj_seed = setup_projectors(projector_kwargs, self.layer_name)
        else:
            self.projector_kwargs = None
            self.proj_dim = None
            self.proj_seed = None

    def _load_chunk(self, chunk_idx: int) -> Tuple[List[Tensor], List[Tensor], ChunkInfo]:
        """Load chunk using HDF5 format."""
        chunk_path = self.cache_dir / f"chunk_{chunk_idx}.h5"
        val1_list, val2_list = [], []

        with h5py.File(chunk_path, 'r') as f:
            # Load chunk info
            info_group = f['info']
            start_idx = info_group.attrs['start_idx']
            size = info_group.attrs['size']

            # Load tensors
            layer_idx = 0
            while f'layer_{layer_idx}' in f:
                layer_group = f[f'layer_{layer_idx}']
                val1 = torch.from_numpy(layer_group['val1'][()]).to(self.device)
                val2 = torch.from_numpy(layer_group['val2'][()]).to(self.device)
                val1_list.append(val1)
                val2_list.append(val2)
                layer_idx += 1

            chunk_info = ChunkInfo(
                start_idx=start_idx,
                size=size,
                layer_shapes=[(v1.shape[1:], v2.shape[1:]) for v1, v2 in zip(val1_list, val2_list)]
            )

        return val1_list, val2_list, chunk_info

    def _process_batch_range(
        self,
        dataloader: torch.utils.data.DataLoader,
        start_batch: int,
        end_batch: int
    ) -> Tuple[List[List[Tensor]], List[List[Tensor]], int]:
        """Process a range of batches."""
        current_val_1 = [None] * len(self.layer_name)
        current_val_2 = [None] * len(self.layer_name)

        num_batches = len(dataloader)
        start_batch = max(0, start_batch)
        end_batch = min(num_batches, end_batch)

        if start_batch >= end_batch:
            return current_val_1, current_val_2, 0

        full_batches = end_batch - start_batch - 1  # number of complete batches
        total_samples = full_batches * dataloader.batch_size

        if end_batch < num_batches:
            total_samples += dataloader.batch_size
        else:
            last_batch_size = len(dataloader.sampler) % dataloader.batch_size
            if last_batch_size == 0:
                last_batch_size = dataloader.batch_size
            total_samples += last_batch_size

        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                desc="Processing batch range...",
                leave=False
            ),
        ):
            if batch_idx < start_batch:
                continue
            if batch_idx >= end_batch:
                break

            input_ids = batch["input_ids"].to(self.device)
            attention_masks = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                labels=labels
            )
            loss = outputs.loss

            pre_acts = [layer.pre_activation for layer in self.layer_name]
            Z_grad = torch.autograd.grad(loss, pre_acts, retain_graph=True)

            with torch.no_grad():
                for layer_id, (layer, z_grad) in enumerate(zip(self.layer_name, Z_grad)):
                    val_1, val_2 = layer.GIP_components(z_grad, per_sample=True)
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

                    if current_val_1[layer_id] is None:
                        current_val_1[layer_id] = torch.zeros((total_samples, *val_1.shape[1:]), device=self.device)
                        current_val_2[layer_id] = torch.zeros((total_samples, *val_2.shape[1:]), device=self.device)

                    col_st = (batch_idx - start_batch) * dataloader.batch_size
                    col_ed = min((batch_idx - start_batch + 1) * dataloader.batch_size, len(dataloader.sampler))
                    current_val_1[layer_id][col_st:col_ed] = val_1
                    current_val_2[layer_id][col_st:col_ed] = val_2

        return current_val_1, current_val_2, total_samples

    def _try_save_chunk(
        self,
        chunk_idx: int,
        val_1_list: List[Tensor],
        val_2_list: List[Tensor],
        start_idx: int
    ) -> bool:
        """Try to save test chunk, return True if successful, False if disk error."""
        try:
            chunk_path = self.cache_dir / f"chunk_{chunk_idx}.h5"
            layer_shapes = [(val1.shape[1:], val2.shape[1:]) for val1, val2 in zip(val_1_list, val_2_list)]
            chunk_info = ChunkInfo(start_idx, val_1_list[0].shape[0], layer_shapes)

            with h5py.File(chunk_path, 'w') as f:
                # Save chunk info
                info_group = f.create_group('info')
                info_group.attrs['start_idx'] = start_idx
                info_group.attrs['size'] = chunk_info.size

                # Save tensors
                for layer_idx, (val1, val2) in enumerate(zip(val_1_list, val_2_list)):
                    layer_group = f.create_group(f'layer_{layer_idx}')
                    layer_group.create_dataset('val1', data=val1.detach().cpu().numpy())
                    layer_group.create_dataset('val2', data=val2.detach().cpu().numpy())

            self.chunks.append(chunk_info)
            return True
        except OSError as e:
            if e.errno == 28:  # No space left on device
                return False
            raise  # Re-raise if it's a different error

    def attribute(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: torch.utils.data.DataLoader,
    ) -> Tensor:
        """Memory-efficient implementation that maximizes disk usage before processing training data."""
        num_train = len(train_dataloader.sampler)
        num_test = len(test_dataloader.sampler)
        grad_dot = torch.zeros(num_train, num_test, device=self.device)

        # Calculate number of batches in test dataloader
        num_test_batches = math.ceil(num_test / test_dataloader.batch_size)
        current_batch = 0

        while current_batch < num_test_batches:
            self.chunks = []  # Reset chunks for new attempt
            self.cleanup()  # Clean up any existing files
            disk_full = False
            batches_in_current_round = 0
            samples_in_current_round = 0

            # Try to fill disk with as many test chunks as possible
            while (current_batch < num_test_batches and
                not disk_full and
                (self.max_test_batches is None or batches_in_current_round < self.max_test_batches)
            ):
                test_chunk_size = self.chunk_size
                chunk_saved = False

                # Calculate remaining batches for this round
                max_batches_this_iteration = (
                    min(
                        num_test_batches - current_batch,
                        self.max_test_batches - batches_in_current_round
                    )
                    if self.max_test_batches is not None
                    else num_test_batches - current_batch
                )

                while test_chunk_size >= self.min_chunk_size and not chunk_saved:
                    try:
                        # Calculate how many batches to process in this iteration
                        batches_to_process = min(
                            math.ceil(test_chunk_size / test_dataloader.batch_size),
                            max_batches_this_iteration
                        )
                        end_batch = current_batch + batches_to_process

                        # Process the range of batches
                        val1_lists, val2_lists, processed_samples = self._process_batch_range(
                            test_dataloader,
                            current_batch,
                            end_batch
                        )

                        if not val1_lists[0].numel():  # Skip if no samples were processed
                            current_batch = num_test_batches  # Exit the outer loop
                            break

                        # Try to save the processed data
                        if self._try_save_chunk(
                            len(self.chunks),
                            val1_lists,
                            val2_lists,
                            current_batch * test_dataloader.batch_size
                        ):
                            batches_in_current_round += batches_to_process
                            samples_in_current_round += processed_samples
                            current_batch = end_batch
                            chunk_saved = True
                        else:
                            test_chunk_size = max(test_chunk_size // 2, self.min_chunk_size)
                            if test_chunk_size == self.min_chunk_size and not chunk_saved:
                                disk_full = True
                                print(f"\nDisk space limit reached:")
                                print(f"- Processed {samples_in_current_round}/{num_test} test samples")
                                print(f"- Processed {batches_in_current_round}/{num_test_batches} test batches")
                                break

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            test_chunk_size = max(test_chunk_size // 2, self.min_chunk_size)
                            if test_chunk_size == self.min_chunk_size and not chunk_saved:
                                disk_full = True
                                print(f"\nGPU memory limit reached:")
                                print(f"- Processed {samples_in_current_round}/{num_test} test samples")
                                print(f"- Processed {batches_in_current_round}/{num_test_batches} test batches")
                                break
                            torch.cuda.empty_cache()
                            continue
                        raise
                # Check if we hit max_test_batches limit
                if (self.max_test_batches is not None and
                    batches_in_current_round >= self.max_test_batches):
                    print(f"\nMaximum batch limit reached:")
                    print(f"- Processed {samples_in_current_round}/{num_test} test samples")
                    print(f"- Processed {batches_in_current_round}/{num_test_batches} test batches")
                    break

            # If we have saved chunks, process training data for all saved chunks
            if self.chunks:  # Only process if we successfully saved some test data
                total_samples = sum(chunk.size for chunk in self.chunks)
                print(f"\nProcessing training data:")
                print(f"- Test samples: {total_samples}/{num_test}")
                print(f"- Test batch range: {current_batch - batches_in_current_round} to {current_batch-1}")
                print(f"- Training samples to process: {num_train}")

                # Process one test chunk at a time
                for chunk_idx, chunk_info in enumerate(tqdm(self.chunks, desc="Processing test chunks", leave=False)):

                    # Load test chunk once
                    test_val_1, test_val_2, _ = self._load_chunk(chunk_idx)

                    # Process all training batches for this test chunk
                    for train_batch_idx, train_batch in enumerate(
                        tqdm(
                            train_dataloader,
                            desc=f"Processing training set for test chunk {chunk_idx+1}/{len(self.chunks)}...",
                            leave=False
                        ),
                    ):
                        train_input_ids = train_batch["input_ids"].to(self.device)
                        train_attention_masks = train_batch["attention_mask"].to(self.device)
                        train_labels = train_batch["labels"].to(self.device)

                        outputs = self.model(
                            input_ids=train_input_ids,
                            attention_mask=train_attention_masks,
                            labels=train_labels
                        )
                        train_loss = outputs.loss

                        train_pre_acts = [layer.pre_activation for layer in self.layer_name]

                        Z_grad_train = torch.autograd.grad(train_loss, train_pre_acts, retain_graph=True)

                        with torch.no_grad():
                            for layer_id, (layer, z_grad_train) in enumerate(zip(self.layer_name, Z_grad_train)):
                                val_1, val_2 = layer.GIP_components(z_grad_train, per_sample=True)
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

                                dLdZ_test, z_test = test_val_1[layer_id], test_val_2[layer_id]
                                dLdZ_train, z_train = val_1, val_2

                                row_st = train_batch_idx * train_dataloader.batch_size
                                row_ed = min((train_batch_idx + 1) * train_dataloader.batch_size, num_train)

                                result = grad_dotprod(dLdZ_train, z_train, dLdZ_test, z_test) *self.lr
                                grad_dot[row_st:row_ed,chunk_info.start_idx:chunk_info.start_idx + chunk_info.size] = result

            # Clean up after processing all chunks
            self.cleanup()

            # Print progress at end of round
            if current_batch < num_test_batches:
                print(f"\nProgress: {current_batch}/{num_test_batches} total batches processed")
                print(f"Starting next round...\n")

        return grad_dot

    def cleanup(self):
        """Remove all cached files."""
        for chunk_idx in range(len(self.chunks)):
            try:
                os.remove(self.cache_dir / f"chunk_{chunk_idx}.h5")
            except FileNotFoundError:
                pass
        self.chunks = []
