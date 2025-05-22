"""
Enhanced base implementation of the Influence Function Attributor with chunked I/O support.
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
    Enhanced base class for Influence Function Attributors with chunked I/O support.
    Implements common functionality with optimized data loading and processing.
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

        # Create appropriate offload strategy with chunking support
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

        # Initialize profiling stats
        self.profiling_stats = ProfilingStats() if self.profile else None

        logger.info(f"Initialized Enhanced BaseAttributor:")
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
        """
        Get the chunk-aligned batch range for a specific worker.

        Args:
            total_batches: Total number of batches in dataset
            worker: Worker specification in format "worker_id/total_workers"

        Returns:
            Tuple of (start_batch, end_batch) that is chunk-aligned

        Raises:
            ValueError: If worker specification is invalid
        """
        try:
            if '/' not in worker:
                raise ValueError("Worker specification must be in format 'worker_id/total_workers'")

            worker_id_str, total_workers_str = worker.split('/')
            worker_id = int(worker_id_str)
            total_workers = int(total_workers_str)

            if worker_id < 0 or total_workers <= 0:
                raise ValueError("worker_id must be >= 0 and total_workers must be > 0")

            if worker_id >= total_workers:
                raise ValueError(f"worker_id ({worker_id}) must be < total_workers ({total_workers})")

        except ValueError as e:
            raise ValueError(f"Invalid worker specification '{worker}': {e}")

        # Calculate total chunks needed
        total_chunks = (total_batches + self.chunk_size - 1) // self.chunk_size

        # Distribute chunks evenly among workers
        chunks_per_worker = total_chunks // total_workers
        remaining_chunks = total_chunks % total_workers

        # Calculate chunk range for this worker
        start_chunk = worker_id * chunks_per_worker + min(worker_id, remaining_chunks)

        if worker_id < remaining_chunks:
            # This worker gets an extra chunk
            end_chunk = start_chunk + chunks_per_worker + 1
        else:
            end_chunk = start_chunk + chunks_per_worker

        # Convert chunk range to batch range
        start_batch = start_chunk * self.chunk_size
        end_batch = min(end_chunk * self.chunk_size, total_batches)

        return start_batch, end_batch

    def _compute_gradients_chunked(
        self,
        dataloader: 'DataLoader',
        is_test: bool = False,
        worker: str = "0/1",
    ) -> Tuple[Dict[int, List[torch.Tensor]], List[int]]:
        """Compute compressed gradients using chunked processing for efficiency."""
        total_batches = len(dataloader)

        # Calculate batch range for the specified worker
        start_batch, end_batch = self._get_worker_batch_range(total_batches, worker)

        desc_prefix = "test" if is_test else "training"
        logger.info(f"Worker {worker}: Computing gradients for {desc_prefix} data")
        logger.info(f"  Batch range: [{start_batch}, {end_batch}) ({end_batch - start_batch} batches)")

        gradients_dict = {}
        batch_sample_counts = []
        using_disk_offload = self.offload == "disk"

        # Create hook manager if not already done
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

        def process_batch_chunk(batch_chunk, chunk_start_idx):
            """Process a chunk of batches efficiently."""
            chunk_gradients = {}
            chunk_sample_counts = []

            for batch_idx_relative, batch in enumerate(batch_chunk):
                # Use global batch index instead of relative index
                batch_idx = chunk_start_idx + batch_idx_relative

                self.model.zero_grad()

                # Prepare inputs
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    batch_size = next(iter(batch.values())).shape[0]
                else:
                    inputs = batch[0].to(self.device)
                    batch_size = batch[0].shape[0]

                chunk_sample_counts.append(batch_size)

                # Forward pass
                if self.profile and self.profiling_stats:
                    torch.cuda.synchronize(self.device)
                    start_time = time.time()

                outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

                if self.profile and self.profiling_stats:
                    torch.cuda.synchronize(self.device)
                    self.profiling_stats.forward += time.time() - start_time

                # Compute loss
                logp = -outputs.loss
                loss = logp - torch.log(1 - torch.exp(logp))

                # Backward pass
                if self.profile and self.profiling_stats:
                    torch.cuda.synchronize(self.device)
                    start_time = time.time()

                loss.backward()

                if self.profile and self.profiling_stats:
                    torch.cuda.synchronize(self.device)
                    self.profiling_stats.backward += time.time() - start_time

                # Get compressed gradients
                with torch.no_grad():
                    compressed_grads = self.hook_manager.get_compressed_grads()

                    batch_grads = []
                    for grad in compressed_grads:
                        if grad is None:
                            batch_grads.append(torch.tensor([]))
                        else:
                            batch_grads.append(grad.detach())

                    # Store gradients using the GLOBAL batch index
                    self.strategy.store_gradients(batch_idx, batch_grads, is_test)

                    # Only store in memory dictionary if NOT using disk offload
                    if is_test or not using_disk_offload:
                        chunk_gradients[batch_idx] = batch_grads
                    else:
                        chunk_gradients[batch_idx] = []

                torch.cuda.empty_cache()

            return chunk_gradients, chunk_sample_counts

        # Skip to start_batch and take only the batches we need
        batches = itertools.islice(dataloader, start_batch, end_batch)
        batch_list = list(batches)

        # Process in fixed chunks (no adaptive sizing)
        for i in tqdm(range(0, len(batch_list), self.chunk_size), desc="Processing batches", unit="chunk"):
            chunk = batch_list[i:i + self.chunk_size]

            # Calculate the global starting batch index for this chunk
            chunk_start_global_idx = start_batch + i

            chunk_grads, chunk_counts = process_batch_chunk(chunk, chunk_start_global_idx)

            gradients_dict.update(chunk_grads)
            batch_sample_counts.extend(chunk_counts)

        # Collect projection time from hook manager if profiling is enabled
        if self.profile and self.profiling_stats and self.hook_manager:
            self.profiling_stats.projection += self.hook_manager.get_compression_time()
            self.profiling_stats.backward -= self.hook_manager.get_compression_time()

        self.strategy.wait_for_async_operations()

        return gradients_dict, batch_sample_counts

    def cache_gradients(self, train_dataloader: 'DataLoader', worker: str = "0/1") -> Dict[int, List[torch.Tensor]]:
        """Cache raw compressed gradients from training data using chunked processing."""
        total_batches = len(train_dataloader)
        start_batch, end_batch = self._get_worker_batch_range(total_batches, worker)

        # Validate and start batch range processing for disk offload
        if self.offload == "disk" and hasattr(self.strategy, 'start_batch_range_processing'):
            self.strategy.start_batch_range_processing(start_batch, end_batch)

        logger.info(f"Worker {worker}: Caching gradients with offload strategy: {self.offload}")
        logger.info(f"  Processing batch range: [{start_batch}, {end_batch}) ({end_batch - start_batch} batches)")

        self.full_train_dataloader = train_dataloader

        if self.sparsifiers is None and self.projectors is None:
            self._setup_compressors(train_dataloader)

        gradients_dict, batch_sample_counts = self._compute_gradients_chunked(
            train_dataloader,
            is_test=False,
            worker=worker
        )

        # Store batch information in metadata
        for i, batch_idx in enumerate(sorted(gradients_dict.keys())):
            sample_idx = batch_idx - start_batch
            if 0 <= sample_idx < len(batch_sample_counts):
                sample_count = batch_sample_counts[sample_idx]
                self.metadata.add_batch_info(batch_idx=batch_idx, sample_count=sample_count)

        self.metadata.save_metadata()

        # Finalize batch range processing for disk offload
        if self.offload == "disk" and hasattr(self.strategy, 'finish_batch_range_processing'):
            self.strategy.finish_batch_range_processing()

            # Wait for async operations to complete
            self.strategy.wait_for_async_operations()

        self._cleanup_hooks()

        logger.info(f"Worker {worker}: Cached gradients for {len(self.layer_names)} modules across {len(gradients_dict)} batches")

        if self.profile and self.profiling_stats:
            return (gradients_dict, self.profiling_stats)
        else:
            return gradients_dict

    @abstractmethod
    def compute_preconditioners(self, damping: Optional[float] = None) -> List[torch.Tensor]:
        """Compute preconditioners (inverse Hessian) from gradients."""
        pass

    @abstractmethod
    def compute_ifvp(self, worker: str = "0/1") -> Dict[int, List[torch.Tensor]]:
        """Compute inverse-Hessian-vector products (IFVP) from gradients and preconditioners."""
        pass

    @abstractmethod
    def compute_self_influence(self, worker: str = "0/1") -> torch.Tensor:
        """Compute self-influence scores for training examples."""
        pass

    @abstractmethod
    def attribute(
        self,
        test_dataloader: 'DataLoader',
        train_dataloader: Optional['DataLoader'] = None,
        use_cached_ifvp: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ProfilingStats]]:
        """Attribute influence of training examples on test examples."""
        pass