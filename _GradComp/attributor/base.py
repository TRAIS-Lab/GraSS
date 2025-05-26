"""
Enhanced Base Attributor with proper worker coordination for metadata.
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

from .strategies import create_offload_strategy, OffloadOptions
from ..core.hook import HookManager
from ..core.metadata import MetadataManager
from ..projection.projector import setup_model_compressors

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
    Enhanced base class for Influence Function Attributors with proper worker coordination.
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

        # Initialize ENHANCED metadata manager
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
            if self.offload == "disk" and hasattr(self.strategy, 'disk_io'):
                self.strategy.disk_io.layer_dims = self.layer_dims
                self.strategy.disk_io.total_proj_dim = self.total_proj_dim

        logger.info(f"Initialized {self.__class__.__name__}:")
        logger.info(f"  Layers: {len(self.layer_names)}, Device: {device}, Offload: {offload}")
        if self.layer_dims:
            logger.info(f"  Dimensions: {len(self.layer_dims)} layers × {self.layer_dims[0] if self.layer_dims else 0} dim = {self.total_proj_dim} total")

    def initialize_dataset_metadata(self, train_dataloader: 'DataLoader') -> None:
        """
        Initialize complete dataset metadata before parallel processing.
        This should be called once before launching parallel workers.
        """
        logger.info("Initializing complete dataset metadata...")
        self.metadata.initialize_full_dataset(train_dataloader)
        self.full_train_dataloader = train_dataloader
        logger.info("Dataset metadata initialization complete")

    def cache_gradients(self, train_dataloader: 'DataLoader', worker: str = "0/1") -> Union[ProcessingInfo, Tuple[ProcessingInfo, ProfilingStats]]:
        """
        Cache gradients using the appropriate storage strategy with proper metadata coordination.

        Args:
            train_dataloader: Training data loader
            worker: Worker specification in format "worker_id/total_workers"

        Returns:
            ProcessingInfo about what was cached, optionally with ProfilingStats
        """
        # Parse worker information
        try:
            worker_id, total_workers = map(int, worker.split('/'))
        except ValueError:
            raise ValueError(f"Invalid worker specification '{worker}'. Use format 'worker_id/total_workers'")

        # Initialize dataset metadata if not already done (for backward compatibility)
        if self.metadata.dataset_info is None:
            logger.warning("Dataset metadata not initialized. Initializing now...")
            self.initialize_dataset_metadata(train_dataloader)

        # Verify we're working with the same dataset
        expected_batches = len(train_dataloader)
        if self.metadata.get_total_batches() != expected_batches:
            logger.warning(f"Batch count mismatch. Expected: {expected_batches}, "
                         f"Metadata: {self.metadata.get_total_batches()}")

        # Calculate worker batch range
        total_batches = self.metadata.get_total_batches()
        start_batch, end_batch = self._get_worker_batch_range(total_batches, worker)

        # Start batch range processing for disk strategy
        if self.offload == "disk" and hasattr(self.strategy, 'start_batch_range_processing'):
            self.strategy.start_batch_range_processing(start_batch, end_batch)

        logger.info(f"Worker {worker}: Caching gradients for batches [{start_batch}, {end_batch}) "
                   f"with offload strategy: {self.offload}")

        self.full_train_dataloader = train_dataloader

        if self.sparsifiers is None and self.projectors is None:
            self._setup_compressors(train_dataloader)

        # Compute and store gradients using the strategy
        _, _ = self._compute_gradients(
            train_dataloader,
            start_batch,
            end_batch,
            is_test=False,
            worker_id=f"{worker_id}"
        )

        # Make sure layer dimensions are saved to metadata
        if self.layer_dims is not None and self.metadata.layer_dims is None:
            self.metadata.set_layer_dims(self.layer_dims)

        # Save metadata (now properly coordinated)
        self.metadata.save_metadata()

        # Finalize batch range processing for disk strategy
        if self.offload == "disk" and hasattr(self.strategy, 'finish_batch_range_processing'):
            self.strategy.finish_batch_range_processing()
            self.strategy.wait_for_async_operations()

        self._cleanup_hooks()

        logger.info(f"Worker {worker}: Completed caching gradients for batches [{start_batch}, {end_batch})")

        # Create processing info
        processing_info = ProcessingInfo(
            num_batches=end_batch - start_batch,
            total_samples=self.metadata.get_total_samples(),  # This is now the FULL dataset
            batch_range=(start_batch, end_batch),
            data_type="gradients"
        )

        if self.profile and self.profiling_stats:
            return (processing_info, self.profiling_stats)
        else:
            return processing_info

    def _compute_gradients(
        self,
        dataloader: 'DataLoader',
        start_batch: int,
        end_batch: int,
        is_test: bool = False,
        worker_id: str = "0"
    ) -> Tuple[Optional[torch.Tensor], Dict[int, Tuple[int, int]]]:
        """
        Compute gradients using efficient subset approach with proper metadata updates.
        """
        # Create efficient subset dataloader for this worker
        worker_dataloader = self._create_worker_dataloader(dataloader, start_batch, end_batch)

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

        # Process batches
        desc = "Computing test gradients" if is_test else f"Computing gradients (Worker {worker_id})"
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
                    logger.info(f"Worker {worker_id}: Detected layer dimensions: {len(self.layer_dims)} layers, total dimension={self.total_proj_dim}")

                    # Save to metadata manager
                    self.metadata.set_layer_dims(self.layer_dims)

                    # Pass to strategy if needed
                    if hasattr(self.strategy, 'layer_dims'):
                        self.strategy.layer_dims = self.layer_dims
                        self.strategy.total_proj_dim = self.total_proj_dim

                # Store gradients using strategy
                self.strategy.store_gradients(global_batch_idx, compressed_grads, is_test)

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

        # Update metadata for training data using the enhanced method
        if not is_test:
            for i, batch_idx in enumerate(range(start_batch, end_batch)):
                if i < len(batch_sample_counts):
                    self.metadata.update_worker_batch_info(batch_idx, batch_sample_counts[i], worker_id)

        # Record compression time from hook manager
        if self.profile and self.profiling_stats and self.hook_manager:
            compression_time = self.hook_manager.get_compression_time()
            self.profiling_stats.compression += compression_time
            logger.debug(f"Worker {worker_id}: Compression time for batch range [{start_batch}, {end_batch}): {compression_time:.3f}s")

        # Return appropriate values
        if is_test:
            if all_gradients:
                return torch.cat(all_gradients, dim=0), batch_mapping
            else:
                return torch.empty(0, self.total_proj_dim), {}
        else:
            return None, {}

    def compute_ifvp(self, worker: str = "0/1") -> Union[ProcessingInfo, Tuple[ProcessingInfo, ProfilingStats]]:
        """
        Compute inverse-Hessian-vector products (IFVP) with proper metadata handling.
        Now works correctly regardless of the number of workers used in cache_gradients.
        """
        logger.info(f"Worker {worker}: Computing IFVP with tensor-based processing")

        # Load batch information - now guaranteed to be complete
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found. Call cache_gradients or initialize_dataset_metadata first.")

        # Verify we have complete dataset info
        if self.metadata.dataset_info is None:
            logger.warning("No complete dataset info found. This may cause issues with partial metadata.")

        # Synchronize layer dimensions
        self._sync_layer_dims()

        if self.layer_dims is None:
            raise ValueError("Layer dimensions not found. Ensure gradients have been computed and stored.")

        # Calculate batch range based on FULL dataset
        total_batches = self.metadata.get_total_batches()
        start_batch, end_batch = self._get_worker_batch_range(total_batches, worker)

        # Start batch range processing
        if self.offload == "disk" and hasattr(self.strategy, 'start_batch_range_processing'):
            self.strategy.start_batch_range_processing(start_batch, end_batch)

        logger.info(f"Processing batch range: [{start_batch}, {end_batch}) out of {total_batches} total batches")

        # Return raw gradients if Hessian type is "none"
        if self.hessian == "none":
            logger.debug("Using raw gradients as IFVP since hessian type is 'none'")
            return self._copy_gradients_as_ifvp(start_batch, end_batch)

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Load all preconditioners once
        preconditioners = []
        for layer_idx in range(len(self.layer_names)):
            precond = self.strategy.retrieve_preconditioner(layer_idx)
            preconditioners.append(precond)

        valid_preconditioners = sum(1 for p in preconditioners if p is not None)
        logger.debug(f"Loaded {valid_preconditioners} preconditioners out of {len(self.layer_names)} layers")

        # Get batch mapping from the COMPLETE metadata
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        processed_batches = 0
        processed_samples = 0

        # Use tensor-based dataloader
        logger.debug("Processing IFVP using tensor-based dataloader")

        dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=4,
            pin_memory=True,
            batch_range=(start_batch, end_batch)
        )

        if dataloader is None:
            logger.error("Failed to create gradient dataloader")
            raise RuntimeError("Cannot create dataloader for IFVP computation")

        for chunk_tensor, batch_mapping in tqdm(dataloader, desc="Computing IFVP from chunks"):
            # Move chunk to device
            chunk_tensor = self.strategy.move_to_device(chunk_tensor)

            # Process each batch in the chunk
            for batch_idx, (start_row, end_row) in batch_mapping.items():
                if batch_idx not in batch_to_sample_mapping:
                    logger.debug(f"Skipping batch {batch_idx} - not in sample mapping")
                    continue

                batch_tensor = chunk_tensor[start_row:end_row]
                batch_ifvp = []

                # Process each layer
                for layer_idx in range(len(self.layer_names)):
                    if preconditioners[layer_idx] is None:
                        batch_ifvp.append(torch.zeros(batch_tensor.shape[0], self.layer_dims[layer_idx]))
                        continue

                    # Extract layer data
                    start_col = sum(self.layer_dims[:layer_idx])
                    end_col = start_col + self.layer_dims[layer_idx]
                    layer_grad = batch_tensor[:, start_col:end_col]

                    if layer_grad.numel() == 0:
                        batch_ifvp.append(torch.zeros(batch_tensor.shape[0], self.layer_dims[layer_idx]))
                        continue

                    # Get preconditioner
                    device_precond = self.strategy.move_to_device(preconditioners[layer_idx])
                    device_precond = device_precond.to(dtype=layer_grad.dtype)

                    # Compute IFVP: H^{-1} @ g
                    ifvp = torch.matmul(device_precond, layer_grad.t()).t()
                    batch_ifvp.append(ifvp)

                    del layer_grad, ifvp, device_precond

                # Store IFVP for this batch using strategy
                self.strategy.store_ifvp(batch_idx, batch_ifvp)

                processed_batches += 1
                processed_samples += batch_tensor.shape[0]

                del batch_ifvp

            del chunk_tensor
            torch.cuda.empty_cache()

        # Finalize batch range processing
        if self.offload == "disk" and hasattr(self.strategy, 'finish_batch_range_processing'):
            self.strategy.finish_batch_range_processing()

        # Clean up
        del preconditioners
        torch.cuda.empty_cache()

        self.strategy.wait_for_async_operations()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

        # Create processing info
        processing_info = ProcessingInfo(
            num_batches=processed_batches,
            total_samples=processed_samples,
            batch_range=(start_batch, end_batch),
            data_type="ifvp"
        )

        if self.profile and self.profiling_stats:
            return (processing_info, self.profiling_stats)
        else:
            return processing_info

    # [Include all other methods from the original BaseAttributor]
    # ... (keeping the rest of the methods the same for now)

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
            elif self.offload == "disk" and hasattr(self.strategy, 'disk_io'):
                self.strategy.disk_io._load_layer_dims_from_metadata()
                if self.strategy.disk_io.layer_dims is not None:
                    self.layer_dims = self.strategy.disk_io.layer_dims
                    self.total_proj_dim = self.strategy.disk_io.total_proj_dim
                    # Also update metadata
                    self.metadata.set_layer_dims(self.layer_dims)
                    logger.debug(f"Loaded layer dimensions from disk IO")

        # Sync to all components
        if self.layer_dims is not None:
            # Update disk IO if needed
            if self.offload == "disk" and hasattr(self.strategy, 'disk_io'):
                self.strategy.disk_io.layer_dims = self.layer_dims
                self.strategy.disk_io.total_proj_dim = self.total_proj_dim

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

    def _create_worker_dataloader(self, dataloader: 'DataLoader', start_batch: int, end_batch: int) -> 'DataLoader':
        """Create an efficient subset dataloader for this worker's batch range."""
        from torch.utils.data import Subset

        dataset = dataloader.dataset
        batch_size = dataloader.batch_size

        # Calculate sample indices for this batch range
        start_idx = start_batch * batch_size
        end_idx = min(end_batch * batch_size, len(dataset))

        # Create subset indices - this is just a list of integers!
        indices = list(range(start_idx, end_idx))
        subset = Subset(dataset, indices)

        # Create new DataLoader with same settings but using the subset
        subset_loader = type(dataloader)(
            subset,
            batch_size=batch_size,
            shuffle=False,  # Important: don't shuffle for worker consistency
            num_workers=0,  # Avoid nested multiprocessing
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
        )

        return subset_loader

    # Abstract methods that subclasses must implement
    @abstractmethod
    def compute_preconditioners(self, damping: Optional[float] = None) -> Union[ProcessingInfo, Tuple[ProcessingInfo, ProfilingStats]]:
        """Compute and store preconditioners from gradients."""
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

    def _copy_gradients_as_ifvp(self, start_batch: int, end_batch: int) -> Union[ProcessingInfo, Tuple[ProcessingInfo, ProfilingStats]]:
        """Copy gradients as IFVP when hessian type is 'none'."""
        # Ensure layer dimensions are loaded
        if self.layer_dims is None:
            self._sync_layer_dims()

        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        processed_batches = 0
        processed_samples = 0

        # Process using tensor dataloader
        dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=1,
            pin_memory=True,
            batch_range=(start_batch, end_batch)
        )

        if dataloader is None:
            logger.error("Failed to create gradient dataloader for copying gradients as IFVP")
            raise RuntimeError("Cannot create dataloader")

        for chunk_tensor, batch_mapping in tqdm(dataloader, desc="Copying gradients as IFVP"):
            for batch_idx, (start_row, end_row) in batch_mapping.items():
                if batch_idx not in batch_to_sample_mapping:
                    continue

                # Extract batch and split into layers
                batch_tensor = chunk_tensor[start_row:end_row]
                gradients = []

                for layer_idx in range(len(self.layer_names)):
                    start_col = sum(self.layer_dims[:layer_idx])
                    end_col = start_col + self.layer_dims[layer_idx]
                    gradients.append(batch_tensor[:, start_col:end_col].contiguous())

                self.strategy.store_ifvp(batch_idx, gradients)

                processed_batches += 1
                processed_samples += batch_tensor.shape[0]

                del gradients

            del chunk_tensor
            torch.cuda.empty_cache()

        # Create processing info
        processing_info = ProcessingInfo(
            num_batches=processed_batches,
            total_samples=processed_samples,
            batch_range=(start_batch, end_batch),
            data_type="ifvp"
        )

        if self.profile and self.profiling_stats:
            return (processing_info, self.profiling_stats)
        else:
            return processing_info