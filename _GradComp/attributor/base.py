"""
Base implementation of the Influence Function Attributor.
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
from ..utils.common import stable_inverse

# Configure logger
logger = logging.getLogger(__name__)

# Type definitions
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
    Base class for Influence Function Attributors.
    Implements common functionality and defines the interface.
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
    ) -> None:
        """
        Initialize the Influence Function Attributor.

        Args:
            setting: Experiment setting/name
            model: PyTorch model
            layer_names: Names of layers to attribute (string or list)
            hessian: Type of Hessian approximation
            damping: Damping factor for Hessian inverse
            profile: Whether to track performance metrics
            device: Device to run computations on
            sparsifier_kwargs: Configuration for sparsifier
            projector_kwargs: Configuration for projector
            offload: Memory management strategy
            cache_dir: Directory for storing data (required for disk offload)
        """
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

        # Create appropriate offload strategy
        self.strategy = create_offload_strategy(
            offload_type=offload,
            device=device,
            layer_names=self.layer_names,
            cache_dir=cache_dir
        )

        # Initialize metadata manager
        self.metadata = MetadataManager(cache_dir or ".", self.layer_names)

        self.full_train_dataloader: Optional['DataLoader'] = None
        self.hook_manager: Optional[HookManager] = None

        self.sparsifiers: Optional[List[Any]] = None
        self.projectors: Optional[List[Any]] = None

        # Initialize profiling stats
        self.profiling_stats = ProfilingStats() if self.profile else None

        # Default chunk size for batch processing
        self.default_chunk_size = 1

        logger.info(f"Initialized BaseAttributor with offload mode: {offload}, device: {device}")

    def _setup_compressors(self, train_dataloader: 'DataLoader') -> None:
        """
        Set up projectors for the model layers

        Args:
            train_dataloader: DataLoader for training data
        """
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

    def _compute_gradients(
        self,
        dataloader: 'DataLoader',
        batch_range: Tuple[int, int],
        is_test: bool = False,
    ) -> Tuple[Dict[int, List[torch.Tensor]], List[int]]:
        """
        Compute compressed gradients for a given dataloader.
        Uses efficient batch range processing with itertools.islice.

        Args:
            dataloader: DataLoader for the data
            batch_range: Tuple of (start_batch, end_batch) to process only a subset of batches
            is_test: Whether this is test data (affects file paths)

        Returns:
            Tuple of (gradients_dict, batch_sample_counts)
            - gradients_dict maps batch_idx to a list of tensors (one per layer) for non-disk offload
            or just contains batch_idx keys with empty lists for disk offload
            - batch_sample_counts contains the number of samples in each batch
        """
        desc_prefix = "test" if is_test else "training"
        start_batch, end_batch = batch_range
        num_batches = end_batch - start_batch

        logger.info(f"Computing gradients for {desc_prefix} data (batches {start_batch} to {end_batch-1})")

        # Initialize storage for gradients (organized by batch)
        gradients_dict = {}
        batch_sample_counts = []

        # Check if using disk offload strategy
        using_disk_offload = self.offload == "disk"

        # Create hook manager if not already done
        if self.hook_manager is None:
            self.hook_manager = HookManager(
                self.model,
                self.layer_names,
                profile=self.profile,
                device=self.device
            )

            # Set sparsifiers in the hook manager
            if self.sparsifiers:
                self.hook_manager.set_sparsifiers(self.sparsifiers)
            # Set projectors in the hook manager
            if self.projectors:
                self.hook_manager.set_projectors(self.projectors)

        # Skip to start_batch and take only the batches we need
        # This avoids iterating through the entire dataloader
        batches = itertools.islice(dataloader, start_batch, end_batch)

        # Process only the selected batches with accurate tqdm
        batch_iterator = tqdm(batches, total=num_batches, desc=f"Computing gradients for {desc_prefix} data")

        # Iterate through the selected batches
        for batch_idx_relative, batch in enumerate(batch_iterator):
            # Calculate the actual batch index
            batch_idx = start_batch + batch_idx_relative

            # Zero gradients
            self.model.zero_grad()

            # Prepare inputs
            if isinstance(batch, dict):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = next(iter(batch.values())).shape[0]
            else:
                inputs = batch[0].to(self.device)
                batch_size = batch[0].shape[0]

            batch_sample_counts.append(batch_size)

            # Forward pass
            if self.profile and self.profiling_stats:
                torch.cuda.synchronize(self.device)
                start_time = time.time()

            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

            if self.profile and self.profiling_stats:
                torch.cuda.synchronize(self.device)
                self.profiling_stats.forward += time.time() - start_time

            # Compute custom loss
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

            # Get compressed gradients from hook manager
            with torch.no_grad():
                compressed_grads = self.hook_manager.get_compressed_grads()

                # Create list of detached gradients for this batch
                batch_grads = []
                for grad in compressed_grads:
                    if grad is None:
                        # Use empty tensor as placeholder for missing gradient
                        batch_grads.append(torch.tensor([]))
                    else:
                        # Detach gradient
                        batch_grads.append(grad.detach())

                # Store gradients using the strategy
                self.strategy.store_gradients(batch_idx, batch_grads, is_test)

                # Only store in memory dictionary if NOT using disk offload
                if not using_disk_offload:
                    gradients_dict[batch_idx] = batch_grads
                else:
                    # For disk offload, just store the batch index in the dict as a marker
                    # but don't keep the actual tensors in memory
                    gradients_dict[batch_idx] = []

            torch.cuda.empty_cache()

        # Collect projection time from hook manager if profiling is enabled
        if self.profile and self.profiling_stats and self.hook_manager:
            self.profiling_stats.projection += self.hook_manager.get_compression_time()
            self.profiling_stats.backward -= self.hook_manager.get_compression_time()

        # Wait for all async operations to complete
        self.strategy.wait_for_async_operations()

        return gradients_dict, batch_sample_counts

    def cache_gradients(
        self,
        train_dataloader: 'DataLoader',
        batch_range: Optional[Tuple[int, int]] = None
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Cache raw compressed gradients from training data.

        Args:
            train_dataloader: DataLoader for the training data
            batch_range: Optional tuple of (start_batch, end_batch) to process only a subset of batches

        Returns:
            Dictionary mapping batch indices to lists of tensors (one tensor per layer)
            For disk offload strategy, may contain empty lists as values
        """
        # Set default batch range if not provided
        if batch_range is None:
            batch_range = (0, len(train_dataloader))

        # Handle batch range
        start_batch, end_batch = batch_range
        batch_msg = f" (processing batches {start_batch} to {end_batch-1})"

        logger.info(f"Caching gradients from training data with offload strategy: {self.offload}{batch_msg}")
        self.full_train_dataloader = train_dataloader

        # Set sparsifiers and projectors if not already done
        if self.sparsifiers is None and self.projectors is None:
            self._setup_compressors(train_dataloader)

        # Compute gradients using common function with batch range
        gradients_dict, batch_sample_counts = self._compute_gradients(
            train_dataloader,
            is_test=False,
            batch_range=batch_range
        )

        # Store batch information in metadata
        start_batch, _ = batch_range
        for i, batch_idx in enumerate(sorted(gradients_dict.keys())):
            # Find the corresponding sample count
            sample_idx = batch_idx - start_batch
            if 0 <= sample_idx < len(batch_sample_counts):
                sample_count = batch_sample_counts[sample_idx]
                # Add this batch to metadata
                self.metadata.add_batch_info(batch_idx=batch_idx, sample_count=sample_count)

        # Save metadata
        self.metadata.save_metadata()

        # Remove hooks after collecting all gradients
        if self.hook_manager:
            self.hook_manager.remove_hooks()
            self.hook_manager = None

        logger.info(f"Cached gradients for {len(self.layer_names)} modules across {len(gradients_dict)} batches")

        if self.profile and self.profiling_stats:
            return (gradients_dict, self.profiling_stats)
        else:
            return gradients_dict

    @abstractmethod
    def compute_preconditioners(self, damping: Optional[float] = None) -> List[torch.Tensor]:
        """
        Compute preconditioners (inverse Hessian) from gradients.

        Args:
            damping: (Adaptive) damping factor for Hessian inverse

        Returns:
            List of preconditioners for each layer
        """
        pass

    @abstractmethod
    def compute_ifvp(
        self,
        batch_range: Optional[Tuple[int, int]] = None
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Compute inverse-Hessian-vector products (IFVP) from gradients and preconditioners.

        Args:
            batch_range: Optional tuple of (start_batch, end_batch) to process only a subset of batches

        Returns:
            Dictionary mapping batch indices to lists of tensors (one tensor per layer)
        """
        pass

    @abstractmethod
    def compute_self_influence(
        self,
        batch_range: Optional[Tuple[int, int]] = None,
        precondition: bool = True,
        damping: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute self-influence scores for training examples.

        Args:
            batch_range: Optional tuple of (start_batch, end_batch) to process only a subset of batches
            precondition: Whether to use preconditioned gradients
            damping: Optional damping parameter (uses self.damping if None)

        Returns:
            Tensor containing self-influence scores for all examples
        """
        pass

    @abstractmethod
    def attribute(
        self,
        test_dataloader: 'DataLoader',
        train_dataloader: Optional['DataLoader'] = None,
        use_cached_ifvp: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ProfilingStats]]:
        """
        Attribute influence of training examples on test examples.

        Args:
            test_dataloader: DataLoader for test data
            train_dataloader: Optional DataLoader for training data if not cached
            use_cached_ifvp: Whether to use cached IFVP (True) or recompute from cached gradients (False)

        Returns:
            Tensor of influence scores (and profiling stats if profile=True)
        """
        pass