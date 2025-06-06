"""
Base Attributor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Tuple, Literal
from dataclasses import dataclass

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    import torch.nn as nn

import torch
from tqdm import tqdm

from ..offload import create_offload_manager, OffloadOptions
from ..core.hook import HookManager
from ..core.metadata import MetadataManager
from ..projection.projector import setup_model_compressors
from ..utils.worker import get_worker_batch_range, create_worker_dataloader

import logging
logger = logging.getLogger(__name__)

HessianOptions = Literal["none", "raw", "kfac", "ekfac"]

@dataclass
class ProfilingStats:
    """Statistics for profiling the algorithm performance."""
    compression: float = 0.0
    forward: float = 0.0
    backward: float = 0.0
    precondition: float = 0.0

@dataclass
class ProcessingInfo:
    """Information about processed data."""
    num_batches: int
    total_samples: int
    batch_range: Tuple[int, int]
    data_type: str  # "gradients", "preconditioners", "ifvp"

class BaseAttributor(ABC):
    """
    Base class for Influence Function Attributors.
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
        self.offload_manager = create_offload_manager(
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
        self.total_proj_dim: Optional[int] = None

        # Initialize profiling stats
        self.profiling_stats = ProfilingStats() if self.profile else None

        # Try to load layer dimensions from metadata if available
        if self.metadata.layer_dims is not None:
            self.layer_dims = self.metadata.layer_dims
            self.total_proj_dim = self.metadata.total_proj_dim
            logger.debug(f"Loaded layer dimensions from metadata")

            # Sync to disk IO if using disk offload
            if self.offload == "disk" and hasattr(self.offload_manager, 'disk_io'):
                self.offload_manager.disk_io.layer_dims = self.layer_dims
                self.offload_manager.disk_io.total_proj_dim = self.total_proj_dim

        logger.info(f"Initialized {self.__class__.__name__}:")
        logger.info(f"  Layers: {len(self.layer_names)}, Device: {device}, Offload: {offload}")
        if self.layer_dims:
            logger.info(f"  Dimensions: {len(self.layer_dims)} layers × {self.layer_dims[0] if self.layer_dims else 0} dim = {self.total_proj_dim} total")

    def _sync_layer_dims(self):
        """Synchronize layer dimensions between components."""
        # Try multiple sources for layer dimensions
        if self.layer_dims is None:
            # Try from metadata first
            if self.metadata.layer_dims is not None:
                self.layer_dims = self.metadata.layer_dims
                self.total_proj_dim = self.metadata.total_proj_dim
                logger.debug(f"Loaded layer dimensions from metadata")

            # Try from disk IO if using disk offload
            elif self.offload == "disk" and hasattr(self.offload_manager, 'disk_io'):
                self.offload_manager.disk_io._load_layer_dims_from_metadata()
                if self.offload_manager.disk_io.layer_dims is not None:
                    self.layer_dims = self.offload_manager.disk_io.layer_dims
                    self.total_proj_dim = self.offload_manager.disk_io.total_proj_dim
                    # Also update metadata
                    self.metadata.set_layer_dims(self.layer_dims)
                    logger.debug(f"Loaded layer dimensions from disk IO")

        # Sync to all components
        if self.layer_dims is not None:
            # Update disk IO if needed
            if self.offload == "disk" and hasattr(self.offload_manager, 'disk_io'):
                self.offload_manager.disk_io.layer_dims = self.layer_dims
                self.offload_manager.disk_io.total_proj_dim = self.total_proj_dim

            # Update metadata if needed
            if self.metadata.layer_dims is None:
                self.metadata.set_layer_dims(self.layer_dims)

            logger.debug(f"Layer dimensions synchronized: {len(self.layer_dims)} layers, total={self.total_proj_dim}")

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

        torch.cuda.empty_cache()

    def _compute_gradients(
        self,
        dataloader: 'DataLoader',
        start_batch: int,
        end_batch: int,
        is_test: bool = False
    ) -> Tuple[Optional[torch.Tensor], Dict[int, Tuple[int, int]]]:
        """
        Compute gradients using efficient subset approach - NO MORE SLOW ISLICE!
        """
        # Create efficient subset dataloader for this worker
        worker_dataloader = create_worker_dataloader(dataloader, start_batch, end_batch)

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

        # For test data, we accumulate in memory and return
        if is_test:
            all_gradients = []
            batch_mapping = {}
            current_row = 0

        batch_sample_counts = []

        # NOW THIS IS FAST! No skipping through thousands of batches
        desc = "Computing test gradients" if is_test else "Computing gradients"
        for batch_idx, batch in enumerate(tqdm(worker_dataloader, desc=desc)):
            # Calculate the global batch index
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

                # Detect layer dimensions on first batch
                if self.layer_dims is None:
                    self.layer_dims = []
                    for grad in compressed_grads:
                        if grad is not None and grad.numel() > 0:
                            self.layer_dims.append(grad.shape[1] if grad.dim() > 1 else grad.numel())
                        else:
                            self.layer_dims.append(0)

                    self.total_proj_dim = sum(self.layer_dims)
                    logger.info(f"Detected layer dimensions: {len(self.layer_dims)} layers, total dimension={self.total_proj_dim}")

                    # Save to metadata manager
                    self.metadata.set_layer_dims(self.layer_dims)

                    # Pass to strategy if needed
                    if hasattr(self.offload_manager, 'layer_dims'):
                        self.offload_manager.layer_dims = self.layer_dims
                        self.offload_manager.total_proj_dim = self.total_proj_dim

                # Store gradients using strategy
                self.offload_manager.store_gradients(global_batch_idx, compressed_grads, is_test)

                # For test data, also accumulate for return
                if is_test:
                    # Concatenate gradients for this batch
                    batch_features = []
                    for grad, dim in zip(compressed_grads, self.layer_dims):
                        if grad is not None and grad.numel() > 0:
                            batch_features.append(grad.cpu())
                        else:
                            batch_features.append(torch.zeros(batch_size, dim))

                    batch_tensor = torch.cat(batch_features, dim=1)
                    all_gradients.append(batch_tensor)
                    batch_mapping[global_batch_idx] = (current_row, current_row + batch_size)
                    current_row += batch_size

            torch.cuda.empty_cache()

        # Update metadata for training data
        if not is_test:
            for i, batch_idx in enumerate(range(start_batch, end_batch)):
                if i < len(batch_sample_counts):
                    self.metadata.add_batch_info(batch_idx=batch_idx, sample_count=batch_sample_counts[i])

        # Record compression time from hook manager
        if self.profile and self.profiling_stats and self.hook_manager:
            compression_time = self.hook_manager.get_compression_time()
            self.profiling_stats.compression += compression_time
            logger.debug(f"Compression time for batch range [{start_batch}, {end_batch}): {compression_time:.3f}s")

        # Return appropriate values
        if is_test:
            if all_gradients:
                return torch.cat(all_gradients, dim=0), batch_mapping
            else:
                return torch.empty(0, self.total_proj_dim), {}
        else:
            return None, {}

    def cache_gradients(self, train_dataloader: 'DataLoader', worker: str = "0/1") -> Union[ProcessingInfo, Tuple[ProcessingInfo, ProfilingStats]]:
        """
        Cache gradients using the appropriate storage strategy.

        Returns:
            ProcessingInfo about what was cached, optionally with ProfilingStats
        """
        # Parse worker information
        try:
            worker_id, total_workers = map(int, worker.split('/'))
        except ValueError:
            raise ValueError(f"Invalid worker specification '{worker}'. Use format 'worker_id/total_workers'")

        # Only worker 0 initializes the complete dataset metadata
        is_master_worker = (worker_id == 0)
        self.metadata.initialize_complete_dataset(train_dataloader, is_master_worker)

        total_batches = len(train_dataloader)
        start_batch, end_batch = get_worker_batch_range(total_batches, self.chunk_size, worker)

        # Start batch range processing for disk strategy
        if self.offload == "disk" and hasattr(self.offload_manager, 'start_batch_range_processing'):
            self.offload_manager.start_batch_range_processing(start_batch, end_batch)

        logger.info(f"Worker {worker}: Caching gradients with offload strategy: {self.offload}")

        self.full_train_dataloader = train_dataloader

        if self.sparsifiers is None and self.projectors is None:
            self._setup_compressors(train_dataloader)

        # Compute and store gradients using the strategy
        _, _ = self._compute_gradients(
            train_dataloader,
            start_batch,
            end_batch,
            is_test=False
        )

        # Make sure layer dimensions are saved to metadata (only master worker saves)
        if self.layer_dims is not None and self.metadata.layer_dims is None:
            self.metadata.set_layer_dims(self.layer_dims)

        # Only master worker saves metadata
        if is_master_worker:
            self.metadata.save_metadata()
            logger.info("Master worker saved complete metadata")

        # Finalize batch range processing for disk strategy
        if self.offload == "disk" and hasattr(self.offload_manager, 'finish_batch_range_processing'):
            self.offload_manager.finish_batch_range_processing()
            self.offload_manager.wait_for_async_operations()

        self._cleanup_hooks()

        logger.info(f"Worker {worker}: Cached gradients for batches [{start_batch}, {end_batch})")

        # Create processing info
        processing_info = ProcessingInfo(
            num_batches=end_batch - start_batch,
            total_samples=self.metadata.get_total_samples(),
            batch_range=(start_batch, end_batch),
            data_type="gradients"
        )

        if self.profile and self.profiling_stats:
            return (processing_info, self.profiling_stats)
        else:
            return processing_info

    @abstractmethod
    def compute_preconditioners(self, damping: Optional[float] = None) -> Union[ProcessingInfo, Tuple[ProcessingInfo, ProfilingStats]]:
        """Compute and store preconditioners from gradients."""
        pass

    @abstractmethod
    def compute_ifvp(self, worker: str = "0/1") -> Union[ProcessingInfo, Tuple[ProcessingInfo, ProfilingStats]]:
        """Compute and store IFVP."""
        pass

    @abstractmethod
    def compute_self_attribution(self, worker: str = "0/1") -> Union[torch.Tensor, Tuple[torch.Tensor, ProfilingStats]]:
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