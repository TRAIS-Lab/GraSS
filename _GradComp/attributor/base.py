"""
Clean base implementation with direct tensor support throughout the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Tuple, Literal
import time
from dataclasses import dataclass
import logging
import gc
import itertools

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    import torch.nn as nn

import torch
from tqdm import tqdm

from .strategies import create_offload_strategy, OffloadOptions
from ..core.hook import HookManager
from ..core.metadata import MetadataManager
from ..projection.projector import setup_model_compressors

logger = logging.getLogger(__name__)

HessianOptions = Literal["none", "raw", "kfac", "ekfac"]

@dataclass
class ProfilingStats:
    """Statistics for profiling the algorithm performance."""
    projection: float = 0.0
    forward: float = 0.0
    backward: float = 0.0
    precondition: float = 0.0
    disk_io: float = 0.0

class BaseAttributor(ABC):
    """
    Clean base class for Influence Function Attributors with direct tensor support.
    """

    def __init__(
        self,
        setting: str,
        model: 'nn.Module',
        layer_names: Union[str, List[str]],
        hessian: HessianOptions = "raw",
        damping: Optional[float] = None,
        profile: bool = False,
        device: str = 'cpu',
        sparsifier_kwargs: Optional[Dict[str, Any]] = None,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        offload: OffloadOptions = "none",
        cache_dir: Optional[str] = None,
        chunk_size: int = 16,
    ) -> None:
        self.setting = setting
        self.model = model
        self.model.to(device)
        self.model.eval()

        # Ensure layer_names is a list
        self.layer_names = [layer_names] if isinstance(layer_names, str) else layer_names

        self.hessian = hessian
        self.damping = damping
        self.profile = profile
        self.device = device
        self.sparsifier_kwargs = sparsifier_kwargs or {}
        self.projector_kwargs = projector_kwargs or {}
        self.offload = offload
        self.chunk_size = chunk_size

        # Create appropriate offload strategy
        self.strategy = create_offload_strategy(
            offload_type=offload,
            device=device,
            layer_names=self.layer_names,
            cache_dir=cache_dir,
            chunk_size=chunk_size
        )

        # Initialize metadata manager
        self.metadata = MetadataManager(cache_dir or ".", self.layer_names)

        self.full_train_dataloader: Optional['DataLoader'] = None
        self.hook_manager: Optional[HookManager] = None

        self.sparsifiers: Optional[List[Any]] = None
        self.projectors: Optional[List[Any]] = None

        # Track layer dimensions
        self.layer_dims: Optional[List[int]] = None

        # Initialize profiling stats
        self.profiling_stats = ProfilingStats() if self.profile else None

        logger.info(f"Initialized Clean BaseAttributor:")
        logger.info(f"  Offload: {offload}, Device: {device}, Chunk size: {chunk_size}")

    def _setup_compressors(self, train_dataloader: 'DataLoader') -> None:
        """Set up projectors for the model layers."""
        logger.debug("Setting up compressors for model layers")
        self.sparsifiers, self.projectors = setup_model_compressors(
            self.model,
            self.layer_names,
            self.sparsifier_kwargs,
            self.projector_kwargs,
            train_dataloader,
            self.setting,
            self.device
        )

    def _cleanup_hooks(self) -> None:
        """Clean up hooks and free memory."""
        if self.hook_manager:
            self.hook_manager.remove_hooks()
            self.hook_manager = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_worker_batch_range(self, total_batches: int, worker: str) -> Tuple[int, int]:
        """Get chunk-aligned batch range for a worker."""
        try:
            worker_id, total_workers = map(int, worker.split('/'))

            if worker_id < 0 or total_workers <= 0 or worker_id >= total_workers:
                raise ValueError("Invalid worker specification")

        except ValueError as e:
            raise ValueError(f"Invalid worker specification '{worker}': {e}")

        # Calculate total chunks
        total_chunks = (total_batches + self.chunk_size - 1) // self.chunk_size

        # Distribute chunks among workers
        chunks_per_worker = total_chunks // total_workers
        remaining_chunks = total_chunks % total_workers

        # Calculate chunk range for this worker
        start_chunk = worker_id * chunks_per_worker + min(worker_id, remaining_chunks)
        end_chunk = start_chunk + chunks_per_worker + (1 if worker_id < remaining_chunks else 0)

        # Convert to batch range
        start_batch = start_chunk * self.chunk_size
        end_batch = min(end_chunk * self.chunk_size, total_batches)

        return start_batch, end_batch

    def _compute_gradients_direct(
        self,
        dataloader: 'DataLoader',
        is_test: bool = False,
        worker: str = "0/1",
    ) -> Tuple[Optional[torch.Tensor], Dict[int, Tuple[int, int]]]:
        """
        Compute gradients and return as concatenated tensor directly.
        Returns:
            - Concatenated gradient tensor of shape (total_samples, total_proj_dim)
            - Mapping from batch_idx to (start_row, end_row) in the tensor
        """
        total_batches = len(dataloader)
        start_batch, end_batch = self._get_worker_batch_range(total_batches, worker)

        desc_prefix = "test" if is_test else "training"
        logger.info(f"Worker {worker}: Computing gradients for {desc_prefix} data")
        logger.info(f"  Batch range: [{start_batch}, {end_batch}) ({end_batch - start_batch} batches)")

        # For test data, accumulate in memory and return tensor
        if is_test:
            return self._compute_test_gradients_as_tensor(dataloader, start_batch, end_batch)

        # For training data, use disk storage
        return self._compute_train_gradients_with_storage(dataloader, start_batch, end_batch)

    def _compute_test_gradients_as_tensor(
        self,
        dataloader: 'DataLoader',
        start_batch: int,
        end_batch: int
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[int, int]]]:
        """Compute test gradients and return as concatenated tensor."""
        # Create hook manager if needed
        if self.hook_manager is None:
            self.hook_manager = HookManager(
                self.model,
                self.layer_names,
                profile=self.profile,
                device=self.device
            )
            if self.sparsifiers:
                self.hook_manager.set_sparsifiers(self.sparsifiers)
            if self.projectors:
                self.hook_manager.set_projectors(self.projectors)

        # Accumulate gradients
        all_gradients = []
        batch_mapping = {}
        current_row = 0

        batches = itertools.islice(dataloader, start_batch, end_batch)

        for batch_idx, batch in enumerate(tqdm(batches, total=end_batch-start_batch, desc="Computing test gradients")):
            global_batch_idx = start_batch + batch_idx

            self.model.zero_grad()

            # Prepare inputs
            if isinstance(batch, dict):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = next(iter(batch.values())).shape[0]
            else:
                inputs = batch[0].to(self.device)
                batch_size = batch[0].shape[0]

            # Forward and backward
            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
            logp = -outputs.loss
            loss = logp - torch.log(1 - torch.exp(logp))
            loss.backward()

            # Get compressed gradients
            with torch.no_grad():
                compressed_grads = self.hook_manager.get_compressed_grads()

                # Detect layer dimensions on first batch
                if self.layer_dims is None:
                    self.layer_dims = []
                    for grad in compressed_grads:
                        if grad is not None and grad.numel() > 0:
                            self.layer_dims.append(grad.shape[1] if grad.dim() > 1 else grad.numel())
                        else:
                            self.layer_dims.append(0)

                # Concatenate gradients for this batch
                batch_features = []
                for grad, dim in zip(compressed_grads, self.layer_dims):
                    if grad is not None and grad.numel() > 0:
                        batch_features.append(grad.cpu())
                    else:
                        batch_features.append(torch.zeros(batch_size, dim))

                batch_tensor = torch.cat(batch_features, dim=1)
                all_gradients.append(batch_tensor)

                # Update mapping
                batch_mapping[global_batch_idx] = (current_row, current_row + batch_size)
                current_row += batch_size

            torch.cuda.empty_cache()

        # Concatenate all batches
        if all_gradients:
            full_tensor = torch.cat(all_gradients, dim=0)
        else:
            total_dim = sum(self.layer_dims) if self.layer_dims else 0
            full_tensor = torch.empty(0, total_dim)

        return full_tensor, batch_mapping

    def _compute_train_gradients_with_storage(
        self,
        dataloader: 'DataLoader',
        start_batch: int,
        end_batch: int
    ) -> Tuple[None, Dict[int, Tuple[int, int]]]:
        """Compute training gradients and store directly to disk."""
        # Create hook manager if needed
        if self.hook_manager is None:
            self.hook_manager = HookManager(
                self.model,
                self.layer_names,
                profile=self.profile,
                device=self.device
            )
            if self.sparsifiers:
                self.hook_manager.set_sparsifiers(self.sparsifiers)
            if self.projectors:
                self.hook_manager.set_projectors(self.projectors)

        batch_sample_counts = []
        batches = itertools.islice(dataloader, start_batch, end_batch)

        for batch_idx, batch in enumerate(tqdm(batches, total=end_batch-start_batch, desc="Computing gradients")):
            global_batch_idx = start_batch + batch_idx

            self.model.zero_grad()

            # Prepare inputs
            if isinstance(batch, dict):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = next(iter(batch.values())).shape[0]
            else:
                inputs = batch[0].to(self.device)
                batch_size = batch[0].shape[0]

            batch_sample_counts.append(batch_size)

            # Forward and backward
            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
            logp = -outputs.loss
            loss = logp - torch.log(1 - torch.exp(logp))
            loss.backward()

            # Get compressed gradients
            with torch.no_grad():
                compressed_grads = self.hook_manager.get_compressed_grads()

                # Convert to list format
                batch_grads = []
                for grad in compressed_grads:
                    if grad is None:
                        batch_grads.append(torch.tensor([]))
                    else:
                        batch_grads.append(grad.detach())

                # Detect layer dimensions on first batch
                if self.layer_dims is None:
                    self.layer_dims = []
                    for grad in batch_grads:
                        if grad.numel() > 0:
                            self.layer_dims.append(grad.shape[1] if grad.dim() > 1 else grad.numel())
                        else:
                            self.layer_dims.append(0)

                    # Pass to disk IO manager
                    if hasattr(self.strategy, 'disk_io'):
                        self.strategy.disk_io.layer_dims = self.layer_dims

                # Store gradients
                self.strategy.store_gradients(global_batch_idx, batch_grads, is_test=False)

            torch.cuda.empty_cache()

        # Update metadata
        for i, batch_idx in enumerate(range(start_batch, end_batch)):
            if i < len(batch_sample_counts):
                self.metadata.add_batch_info(batch_idx=batch_idx, sample_count=batch_sample_counts[i])

        return None, {}

    def cache_gradients(self, train_dataloader: 'DataLoader', worker: str = "0/1") -> None:
        """Cache gradients using direct tensor storage."""
        total_batches = len(train_dataloader)
        start_batch, end_batch = self._get_worker_batch_range(total_batches, worker)

        # Start batch range processing
        if self.offload == "disk" and hasattr(self.strategy, 'start_batch_range_processing'):
            self.strategy.start_batch_range_processing(start_batch, end_batch)

        logger.info(f"Worker {worker}: Caching gradients with offload strategy: {self.offload}")

        self.full_train_dataloader = train_dataloader

        if self.sparsifiers is None and self.projectors is None:
            self._setup_compressors(train_dataloader)

        # Compute and store gradients
        self._compute_train_gradients_with_storage(train_dataloader, start_batch, end_batch)

        self.metadata.save_metadata()

        # Finalize batch range processing
        if self.offload == "disk" and hasattr(self.strategy, 'finish_batch_range_processing'):
            self.strategy.finish_batch_range_processing()
            self.strategy.wait_for_async_operations()

        self._cleanup_hooks()

        logger.info(f"Worker {worker}: Cached gradients for {len(self.layer_names)} modules")

    @abstractmethod
    def compute_preconditioners(self, damping: Optional[float] = None) -> List[torch.Tensor]:
        """Compute preconditioners from gradients."""
        pass

    @abstractmethod
    def compute_ifvp(self, worker: str = "0/1") -> None:
        """Compute and store IFVP."""
        pass

    @abstractmethod
    def compute_self_influence(self, worker: str = "0/1") -> torch.Tensor:
        """Compute self-influence scores."""
        pass

    @abstractmethod
    def attribute(
        self,
        test_dataloader: 'DataLoader',
        train_dataloader: Optional['DataLoader'] = None,
        use_cached_ifvp: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ProfilingStats]]:
        """Attribute influence scores."""
        pass