from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Literal, List, Optional, Union, Tuple, TypedDict, cast
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import glob
import contextlib

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    import torch.nn as nn

import torch
from tqdm import tqdm

from .hook import HookManager
from .utils import stable_inverse
from .projector import setup_model_projectors

# Type hints
TensorOrPath = Union[torch.Tensor, str]
OffloadOptions = Literal["none", "cpu", "disk"]
HessianOptions = Literal["none", "raw", "kfac", "ekfac"]
DataTypeOptions = Literal["gradients", "preconditioners", "ifvp"]


@dataclass
class ProfilingStats:
    """Statistics for profiling the algorithm performance."""
    projection: float = 0.0
    forward: float = 0.0
    backward: float = 0.0
    precondition: float = 0.0
    disk_io: float = 0.0


@dataclass
class BatchInfo:
    """Data structure for batch processing information."""
    batch_range: Tuple[int, int]
    sample_counts: List[int]
    total_samples: int


@contextlib.contextmanager
def async_stream():
    """
    Context manager for asynchronous CUDA stream operations.
    All operations within this context will be executed asynchronously.
    """
    if torch.cuda.is_available():
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            yield stream
        # Ensure stream operations complete before returning
        stream.synchronize()
    else:
        yield None



class MetadataManager:
    """
    Manages metadata about batches, layers, and processing state.
    """

    def __init__(self, cache_dir: str, layer_names: List[str]):
        """
        Initialize the MetadataManager.

        Args:
            cache_dir: Directory for metadata storage
            layer_names: Names of neural network layers
        """
        self.cache_dir = cache_dir
        self.layer_names = layer_names
        self.batch_info = {}

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

            # Check for existing metadata
            self._load_metadata_if_exists()

    def _get_metadata_path(self) -> str:
        """Get the path to the metadata file."""
        return os.path.join(self.cache_dir, "batch_metadata.json")

    def _load_metadata_if_exists(self) -> None:
        """Load metadata from disk if it exists."""
        metadata_path = self._get_metadata_path()
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    # Convert string keys back to integers
                    self.batch_info = {
                        int(batch_idx): BatchInfo(**info)
                        for batch_idx, info in metadata['batch_info'].items()
                    }
            except Exception as e:
                print(f"Error loading metadata: {e}")

    def save_batch_info(self, batch_idx: int, batch_range: Tuple[int, int],
                       sample_counts: List[int], total_samples: int) -> None:
        """
        Save information about a processed batch.

        Args:
            batch_idx: Index of the batch
            batch_range: Tuple of (start_batch, end_batch)
            sample_counts: Sample counts for each mini-batch
            total_samples: Total number of samples in the batch
        """
        self.batch_info[batch_idx] = BatchInfo(
            batch_range=batch_range,
            sample_counts=sample_counts,
            total_samples=total_samples
        )

        # Save to disk if cache directory is set
        if self.cache_dir:
            metadata_path = self._get_metadata_path()

            # Convert to serializable format
            serializable_info = {
                str(idx): {
                    "batch_range": info.batch_range,
                    "sample_counts": info.sample_counts,
                    "total_samples": info.total_samples
                }
                for idx, info in self.batch_info.items()
            }

            metadata = {
                'batch_info': serializable_info,
                'layer_names': self.layer_names,
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def get_batch_indices(self) -> List[int]:
        """Get all batch indices."""
        return sorted(self.batch_info.keys())

    def get_total_samples(self) -> int:
        """Get total number of samples across all batches."""
        return sum(info.total_samples for _, info in self.batch_info.items())

    def get_batch_to_sample_mapping(self) -> Dict[int, Tuple[int, int]]:
        """
        Get mapping from batch indices to sample ranges.

        Returns:
            Dictionary mapping batch index to (start_sample, end_sample) tuple
        """
        mapping = {}
        current_sample = 0

        for batch_idx in sorted(self.batch_info.keys()):
            info = self.batch_info[batch_idx]
            mapping[batch_idx] = (current_sample, current_sample + info.total_samples)
            current_sample += info.total_samples

        return mapping


class IFAttributor:
    """
    Influence function calculator using hooks for efficient gradient projection.
    Works with standard PyTorch layers with support for different offloading strategies.

    Uses a batch-oriented file structure for more efficient I/O:
    - Gradient and IFVP data are grouped by batch (all layers in one file)
    - Preconditioners are stored per-layer
    """

    def __init__(
        self,
        setting: str,
        model: nn.Module,
        layer_names: Union[str, List[str]],
        hessian: HessianOptions = "raw",
        damping: Optional[float] = None,
        profile: bool = False,
        device: str = 'cpu',
        projector_kwargs: Optional[Dict[str, Any]] = None,
        offload: OffloadOptions = "none",
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Influence Function Attributor.

        Args:
            setting: The setting of the experiment. Used for logging and locating the directory.
            model: PyTorch model.
            layer_names: Names of layers to attribute. Can be a string or list of strings.
            hessian: Type of Hessian approximation. Defaults to "raw".
            damping: Damping used when calculating the Hessian inverse. Defaults to None.
            profile: Record time used in various parts of the algorithm run. Defaults to False.
            device: Device to run the model on. Defaults to 'cpu'.
            projector_kwargs: Keyword arguments for projector. Defaults to None.
            offload: Memory management strategy. Defaults to "none".
            cache_dir: Directory to save files. Only used when offload="disk". Defaults to None.
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
        self.projector_kwargs = projector_kwargs or {}
        self.offload = offload
        self.cpu_offload = offload in ["cpu", "disk"]

        # Initialize disk I/O manager and metadata manager if using disk offload
        if offload == "disk":
            if cache_dir is None:
                raise ValueError("Cache directory must be provided when using disk offload")
            self.cache_dir = cache_dir
            self.disk_io = DiskIOManager(cache_dir, setting, hessian=hessian)
            self.metadata = MetadataManager(cache_dir, self.layer_names)
        else:
            self.cache_dir = None
            self.disk_io = None
            self.metadata = None

        self.full_train_dataloader: Optional[DataLoader] = None
        self.hook_manager: Optional[HookManager] = None
        self.projectors: Optional[List[Any]] = None

        # For in-memory storage
        self.cached_gradients: Optional[Dict[int, List[torch.Tensor]]] = None
        self.cached_ifvp: Optional[Dict[int, List[torch.Tensor]]] = None
        self.preconditioners: Optional[List[torch.Tensor]] = None

        # Initialize profiling stats
        self.profiling_stats = ProfilingStats() if self.profile else None

    def _setup_projectors(self, train_dataloader: DataLoader) -> None:
        """
        Set up projectors for the model layers

        Args:
            train_dataloader: DataLoader for training data
        """
        if not self.projector_kwargs:
            self.projectors = []
            return

        self.projectors = setup_model_projectors(
            self.model,
            self.layer_names,
            self.projector_kwargs,
            train_dataloader,
            self.setting,
            self.device
        )

    def _compute_gradients(
        self,
        dataloader: DataLoader,
        batch_range: Tuple[int, int],
        is_test: bool = False,
    ) -> Tuple[Dict[int, List[torch.Tensor]], List[int]]:
        """
        Compute projected gradients for a given dataloader.

        Args:
            dataloader: DataLoader for the data
            batch_range: Tuple of (start_batch, end_batch) to process only a subset of batches
            is_test: Whether this is test data (affects file paths)

        Returns:
            Tuple of (gradients_dict, batch_sample_counts)
            - gradients_dict maps batch_idx to a list of tensors (one per layer)
            - batch_sample_counts contains the number of samples in each batch
        """
        if is_test:
            desc = f"Computing gradients for test data"
        else:
            desc = f"Computing gradients for training data"

        start_batch, end_batch = batch_range
        desc += f" (batches {start_batch} to {end_batch-1})"

        # Create a list of batches to process
        batch_indices = list(range(start_batch, end_batch))
        # Create description that reflects the actual work
        actual_desc = f"{desc} ({len(batch_indices)} batches)"

        # Initialize storage for gradients (organized by batch)
        gradients_dict = {}
        batch_sample_counts = []

        # Create hook manager if not already done
        if self.hook_manager is None:
            self.hook_manager = HookManager(
                self.model,
                self.layer_names,
            )

            # Set projectors in the hook manager
            if self.projectors:
                self.hook_manager.set_projectors(self.projectors)

        # Prepare to collect batches
        selected_batches = []

        # First iterate through the dataloader to collect the batches we need
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx in batch_indices:
                selected_batches.append((batch_idx, batch))
            if batch_idx >= end_batch - 1:
                break  # No need to iterate further

        # Now process only the selected batches with accurate tqdm
        batch_iterator = tqdm(selected_batches, desc=actual_desc)

        # Iterate through the selected batches
        for batch_idx, batch in batch_iterator:
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

            # Get projected gradients from hook manager
            with torch.no_grad():
                projected_grads = self.hook_manager.get_projected_grads()

                # Create list of detached gradients for this batch
                batch_grads = []
                for idx, grad in enumerate(projected_grads):
                    if grad is None:
                        # Use empty tensor as placeholder for missing gradient
                        batch_grads.append(torch.tensor([]))
                    else:
                        # Detach gradient and move to appropriate device
                        detached_grad = grad.detach()
                        if self.offload == "cpu" or self.offload == "disk":
                            batch_grads.append(detached_grad.cpu())
                            del detached_grad
                        else:
                            batch_grads.append(detached_grad)

                # Store gradients for this batch
                gradients_dict[batch_idx] = batch_grads

                # For disk offload, save the batch immediately
                if self.offload == "disk":
                    # Create dictionary with layer indices as keys
                    grad_dict = {idx: grad for idx, grad in enumerate(batch_grads)}

                    # Save to disk
                    file_path = self.disk_io.get_path(
                        data_type='gradients',
                        batch_idx=batch_idx,
                        is_test=is_test
                    )
                    self.disk_io.save_dict(grad_dict, file_path)

                    # Free memory
                    del batch_grads

            # GPU memory management - ensure we don't run out of memory
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        # Wait for all disk operations to complete if using disk offload
        if self.offload == "disk":
            self.disk_io.wait_for_async_operations()

        return gradients_dict, batch_sample_counts

    def cache_gradients(
        self,
        train_dataloader: DataLoader,
        save: bool = False,
        batch_range: Optional[Tuple[int, int]] = None,
        batch_idx: Optional[int] = None
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Cache raw projected gradients from training data to disk or memory.
        Organizes gradients by batch, with each batch file containing all layer gradients.

        Args:
            train_dataloader: DataLoader for the training data
            save: Whether to force saving gradients (already saved for disk offload)
            batch_range: Optional tuple of (start_batch, end_batch) to process only a subset of batches
            batch_idx: Optional identifier for the batch

        Returns:
            Dictionary mapping batch indices to lists of tensors (one tensor per layer)
        """
        # If batch_range and batch_idx are not provided, process the entire dataset
        if batch_range is None and batch_idx is None:
            batch_range = (0, len(train_dataloader))
            batch_idx = 0
            print(f"No batch information provided, processing entire dataset as batch_idx={batch_idx}")

        # Handle batch range
        batch_msg = ""
        if batch_range is not None:
            start_batch, end_batch = batch_range
            batch_msg = f" (processing batches {start_batch} to {end_batch})"

        print(f"Caching gradients from training data with offload strategy: {self.offload}{batch_msg}...")
        self.full_train_dataloader = train_dataloader

        # Set up the projectors if not already done
        if self.projectors is None:
            self._setup_projectors(train_dataloader)

        # Compute gradients using common function with batch range
        gradients_dict, batch_sample_counts = self._compute_gradients(
            train_dataloader,
            is_test=False,
            batch_range=batch_range
        )

        # Store metadata about this batch processing
        if self.metadata and batch_idx is not None:
            self.metadata.save_batch_info(
                batch_idx=batch_idx,
                batch_range=batch_range,
                sample_counts=batch_sample_counts,
                total_samples=sum(batch_sample_counts)
            )

        # Store gradients in memory if not using disk offload
        if self.offload != "disk":
            self.cached_gradients = gradients_dict

        print(f"Cached gradients for {len(self.layer_names)} modules across {len(gradients_dict)} batches")

        # Remove hooks after collecting all gradients
        if self.hook_manager:
            self.hook_manager.remove_hooks()
            self.hook_manager = None

        if self.profile and self.profiling_stats:
            return (gradients_dict, self.profiling_stats)
        else:
            return gradients_dict

    def compute_preconditioners(self, damping: Optional[float] = None, save: bool = True) -> List[torch.Tensor]:
        """
        Compute preconditioners (inverse Hessian) from gradients based on the specified Hessian type.
        Accumulates Hessian contributions from all batches to compute a single preconditioner per layer.
        Stores preconditioners in individual files per layer.

        Args:
            damping: Damping factor for Hessian inverse (uses self.damping if None)
            save: Whether to save preconditioners to disk (for disk offload)

        Returns:
            List of preconditioners for each layer (one preconditioner per layer)
        """
        print(f"Computing preconditioners with hessian type: {self.hessian}...")

        # Use instance damping if not provided
        if damping is None:
            damping = self.damping

        # Check if we need to load batch information from metadata
        if self.metadata and not self.metadata.batch_info:
            print("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not hasattr(self, 'metadata') or not self.metadata.batch_info:
            raise ValueError("No batch information found. Call cache_gradients first.")

        # If hessian type is "none", no preconditioners needed
        if self.hessian == "none":
            print("Hessian type is 'none', skipping preconditioner computation")
            self.preconditioners = [None] * len(self.layer_names)
            return self.preconditioners

        # Calculate total samples across all batches
        total_samples = self.metadata.get_total_samples()
        print(f"Total samples across all batches: {total_samples}")

        # Calculate Hessian for each layer (one per layer, not per batch)
        preconditioners: List[Optional[torch.Tensor]] = [None] * len(self.layer_names)

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Process each layer
        for layer_idx, layer_name in tqdm(enumerate(self.layer_names), desc="Processing layers", total=len(self.layer_names)):
            # Initialize Hessian accumulator on GPU for efficient computation
            hessian_accumulator = None
            sample_count = 0

            # Process based on offload strategy
            if self.offload == "disk":
                # Find all gradient batch files
                batch_files = self.disk_io.find_batch_files('gradients')

                # Process in chunks to avoid memory issues
                chunk_size = 10  # Adjust based on memory constraints
                for i in range(0, len(batch_files), chunk_size):
                    chunk_files = batch_files[i:i+chunk_size]

                    # Load gradient dictionaries in parallel
                    batch_dicts = self.disk_io.batch_load_dicts(chunk_files)

                    for batch_dict in batch_dicts:
                        # Skip if this layer doesn't have gradients in this batch
                        if layer_idx not in batch_dict or batch_dict[layer_idx].numel() == 0:
                            continue

                        # Get gradient for this layer
                        grad = batch_dict[layer_idx].to(self.device)

                        # Accumulate batch contribution to Hessian
                        batch_hessian = torch.matmul(grad.t(), grad)

                        # Initialize or update accumulator
                        if hessian_accumulator is None:
                            hessian_accumulator = batch_hessian
                        else:
                            hessian_accumulator += batch_hessian

                        # Update sample count
                        sample_count += grad.shape[0]

                        # Clean up to save memory
                        del grad, batch_hessian
                        torch.cuda.empty_cache()
            else:
                # For in-memory processing, iterate through batch indices
                if self.cached_gradients:
                    for batch_idx, batch_grads in self.cached_gradients.items():
                        # Skip if layer_idx is out of range or gradient is empty
                        if layer_idx >= len(batch_grads) or batch_grads[layer_idx].numel() == 0:
                            continue

                        grad = batch_grads[layer_idx]

                        if self.offload == "cpu":
                            # Move to GPU for computation
                            grad = grad.to(self.device)

                        # Compute contribution to Hessian
                        batch_hessian = torch.matmul(grad.t(), grad)

                        # Initialize or update accumulator
                        if hessian_accumulator is None:
                            hessian_accumulator = batch_hessian
                        else:
                            hessian_accumulator += batch_hessian

                        # Update sample count
                        sample_count += grad.shape[0]

                        # Clean up to save memory if needed
                        if self.offload == "cpu":
                            del grad, batch_hessian
                            torch.cuda.empty_cache()

            # If we have accumulated Hessian, compute preconditioner
            if hessian_accumulator is not None:
                # Normalize by total number of samples
                hessian = hessian_accumulator / sample_count

                # Compute inverse based on Hessian type
                if self.hessian == "raw":
                    precond = stable_inverse(hessian, damping=damping)

                    # Save or store based on offload strategy
                    if self.offload == "disk" and save:
                        # Save to disk with async stream
                        with async_stream():
                            cpu_precond = precond.cpu()
                            file_path = self.disk_io.get_path('preconditioners', layer_idx=layer_idx)
                            self.disk_io.save_tensor(cpu_precond, file_path)
                            preconditioners[layer_idx] = cpu_precond
                    else:
                        # Store in memory
                        if self.cpu_offload:
                            with async_stream():
                                preconditioners[layer_idx] = precond.cpu()
                        else:
                            preconditioners[layer_idx] = precond

                    # Clean up
                    del precond

                elif self.hessian in ["kfac", "ekfac"]:
                    # Store Hessian itself for KFAC-type preconditioners
                    if self.offload == "disk" and save:
                        file_path = self.disk_io.get_path('preconditioners', layer_idx=layer_idx)
                        self.disk_io.save_tensor(hessian.cpu(), file_path)
                        preconditioners[layer_idx] = hessian.cpu()
                    else:
                        # Store in memory
                        preconditioners[layer_idx] = hessian.cpu() if self.cpu_offload else hessian

                # Clean up
                del hessian_accumulator, hessian
                torch.cuda.empty_cache()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

        # Store preconditioners
        self.preconditioners = preconditioners

        # Wait for async operations to complete
        if self.offload == "disk":
            self.disk_io.wait_for_async_operations()

        if self.profile and self.profiling_stats:
            return (preconditioners, self.profiling_stats)
        else:
            return preconditioners

    def compute_ifvp(self, save: bool = True) -> Dict[int, List[torch.Tensor]]:
        """
        Compute inverse-Hessian-vector products (IFVP) from gradients and preconditioners.
        Organizes IFVP by batch, with each batch file containing all layer IFVPs.

        Args:
            save: Whether to save IFVPs to disk (for disk offload)

        Returns:
            Dictionary mapping batch indices to lists of tensors (one tensor per layer)
        """
        print("Computing inverse-Hessian-vector products (IFVP)...")

        # Load batch information if needed
        if self.metadata and not self.metadata.batch_info:
            print("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not hasattr(self, 'metadata') or not self.metadata.batch_info:
            raise ValueError("No batch information found. Call cache_gradients first.")

        # Return raw gradients if Hessian type is "none"
        if self.hessian == "none":
            print("Using raw gradients as IFVP since hessian type is 'none'")

            if self.offload == "disk":
                # For disk offload, create copies of gradient files as IFVP files
                ifvp_dict = {}

                # Find all gradient batch files
                batch_files = self.disk_io.find_batch_files('gradients')

                for file_path in batch_files:
                    batch_idx = self.disk_io.extract_batch_idx(file_path)

                    # Load gradient dictionary
                    grad_dict = self.disk_io.load_dict(file_path)

                    # Save as IFVP
                    ifvp_path = self.disk_io.get_path('ifvp', batch_idx=batch_idx)
                    self.disk_io.save_dict(grad_dict, ifvp_path)

                    # Add to result (won't be used if offload is disk)
                    ifvp_dict[batch_idx] = [grad_dict[i] if i in grad_dict else torch.tensor([])
                                         for i in range(len(self.layer_names))]

                    # Clean up
                    del grad_dict

                # Store in memory (not needed for disk offload, but for consistency)
                self.cached_ifvp = ifvp_dict
                return ifvp_dict
            else:
                # For memory storage, use cached gradients directly
                self.cached_ifvp = self.cached_gradients
                return self.cached_gradients

        # Initialize result structure
        ifvp_dict = {}

        # Get preconditioners - either from memory or disk
        preconditioners = self._get_preconditioners()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Process based on offload strategy
        if self.offload == "disk":
            # Find all gradient batch files
            batch_files = self.disk_io.find_batch_files('gradients')

            # Process in chunks to avoid memory issues
            chunk_size = 5  # Adjust based on memory constraints
            for i in range(0, len(batch_files), chunk_size):
                chunk_files = batch_files[i:i+chunk_size]

                # Process each file in the chunk
                for file_path in chunk_files:
                    batch_idx = self.disk_io.extract_batch_idx(file_path)

                    # Load gradient dictionary
                    grad_dict = self.disk_io.load_dict(file_path)

                    # Create dictionary for IFVP results
                    ifvp_batch_dict = {}

                    # Compute IFVP for each layer in this batch
                    for layer_idx, precond in enumerate(preconditioners):
                        # Skip if preconditioner is None or layer not in gradient dictionary
                        if precond is None or layer_idx not in grad_dict or grad_dict[layer_idx].numel() == 0:
                            ifvp_batch_dict[layer_idx] = torch.tensor([])
                            continue

                        # Load preconditioner to device if it's a string (file path)
                        if isinstance(precond, str):
                            precond_tensor = self.disk_io.load_tensor(precond).to(self.device)
                        else:
                            precond_tensor = precond.to(self.device)

                        # Load gradient to device
                        grad = grad_dict[layer_idx].to(self.device)

                        # Compute IFVP
                        result = torch.matmul(precond_tensor, grad.t()).t()

                        # Store in dictionary
                        ifvp_batch_dict[layer_idx] = result.cpu()

                        # Clean up
                        del grad, precond_tensor, result
                        torch.cuda.empty_cache()

                    # Save IFVP batch to disk
                    ifvp_path = self.disk_io.get_path('ifvp', batch_idx=batch_idx)
                    self.disk_io.save_dict(ifvp_batch_dict, ifvp_path)

                    # Store in result dictionary (won't be used if offload is disk)
                    ifvp_dict[batch_idx] = [ifvp_batch_dict[i] if i in ifvp_batch_dict else torch.tensor([])
                                        for i in range(len(self.layer_names))]

                    # Clean up
                    del grad_dict, ifvp_batch_dict
                    torch.cuda.empty_cache()
        else:
            # For in-memory processing
            if self.cached_gradients:
                for batch_idx, batch_grads in self.cached_gradients.items():
                    # List to store IFVP for each layer in this batch
                    batch_ifvp = []

                    # Process each layer
                    for layer_idx, precond in enumerate(preconditioners):
                        # Skip if preconditioner is None or layer_idx is out of range
                        if precond is None or layer_idx >= len(batch_grads) or batch_grads[layer_idx].numel() == 0:
                            batch_ifvp.append(torch.tensor([]))
                            continue

                        # Get gradient
                        grad = batch_grads[layer_idx]

                        # Move tensors to GPU if needed
                        if self.offload == "cpu":
                            grad = grad.to(self.device)
                            if isinstance(precond, str):
                                precond_tensor = torch.load(precond).to(self.device)
                            else:
                                precond_tensor = precond.to(self.device)
                        else:
                            precond_tensor = precond

                        # Compute IFVP
                        result = torch.matmul(precond_tensor, grad.t()).t()

                        # Add to results
                        if self.offload == "cpu":
                            batch_ifvp.append(result.cpu())
                            del result
                            torch.cuda.empty_cache()
                        else:
                            batch_ifvp.append(result)

                    # Store batch IFVP
                    ifvp_dict[batch_idx] = batch_ifvp

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

        # Store IFVP results
        self.cached_ifvp = ifvp_dict

        # Wait for async operations to complete
        if self.offload == "disk":
            self.disk_io.wait_for_async_operations()

        if self.profile and self.profiling_stats:
            return (ifvp_dict, self.profiling_stats)
        else:
            return ifvp_dict

    def attribute(
        self,
        test_dataloader: DataLoader,
        train_dataloader: Optional[DataLoader] = None,
        use_cached_ifvp: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ProfilingStats]]:
        """
        Attribute influence of training examples on test examples.
        Works with batched training gradients.

        Args:
            test_dataloader: DataLoader for test data
            train_dataloader: Optional DataLoader for training data if not cached
            use_cached_ifvp: Whether to use cached IFVP (True) or recompute from cached gradients (False)

        Returns:
            Tensor of influence scores (and profiling stats if profile=True)
        """
        # Load batch information if needed
        if self.metadata and not self.metadata.batch_info:
            print("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not hasattr(self, 'metadata') or not self.metadata.batch_info:
            if train_dataloader is None:
                raise ValueError("No batch information found and no training dataloader provided.")

            # Cache gradients if needed
            print("No batch metadata found. Caching gradients from provided dataloader...")
            self.cache_gradients(train_dataloader)

        # Validate input
        if train_dataloader is None and self.full_train_dataloader is None and not self.metadata.batch_info:
            raise ValueError("No training data provided or cached.")

        # Set up the projectors if not already done
        if self.projectors is None and test_dataloader is not None:
            self._setup_projectors(test_dataloader)

        # Get IFVP - either from memory, disk, or compute new
        ifvp_dict = self._get_ifvp(use_cached_ifvp)

        # Compute test gradients
        test_grads_dict, _ = self._compute_gradients(
            test_dataloader,
            is_test=True,
            batch_range=(0, len(test_dataloader))
        )

        torch.cuda.empty_cache()

        # Calculate total training samples and map batches to sample indices
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        total_train_samples = self.metadata.get_total_samples()

        # Initialize influence scores in memory
        num_test = len(test_dataloader.sampler)
        IF_score = torch.zeros(total_train_samples, num_test, device="cpu")

        # Map test batch indices to sample ranges
        test_batch_indices = {}
        current_index = 0
        for test_batch_idx in sorted(test_grads_dict.keys()):
            batch_grads = test_grads_dict[test_batch_idx][0]  # Use first layer to get batch size
            batch_size = batch_grads.shape[0] if batch_grads.numel() > 0 else 0
            test_batch_indices[test_batch_idx] = (current_index, current_index + batch_size)
            current_index += batch_size

        # Remove hooks after collecting test gradients
        if self.hook_manager:
            self.hook_manager.remove_hooks()
            self.hook_manager = None

        # Process each test batch
        for test_batch_idx in sorted(test_grads_dict.keys()):
            # Get column indices for this test batch
            col_st, col_ed = test_batch_indices[test_batch_idx]

            # Load test gradients for this batch
            if self.offload == "disk":
                # Load test batch from disk
                test_path = self.disk_io.get_path('gradients', batch_idx=test_batch_idx, is_test=True)
                test_dict = self.disk_io.load_dict(test_path)
                test_grads = [test_dict[i] if i in test_dict else torch.tensor([])
                           for i in range(len(self.layer_names))]
            else:
                # Get from memory
                test_grads = test_grads_dict[test_batch_idx]

            # Process each training batch
            train_batch_indices = sorted(batch_to_sample_mapping.keys())
            chunk_size = 5  # Adjust based on memory constraints

            for i in range(0, len(train_batch_indices), chunk_size):
                batch_chunk = train_batch_indices[i:i+chunk_size]

                # Process each batch in the chunk
                for train_batch_idx in batch_chunk:
                    # Get row indices for this train batch
                    row_st, row_ed = batch_to_sample_mapping[train_batch_idx]

                    # Load IFVP for this batch
                    if self.offload == "disk":
                        # Load from disk
                        ifvp_path = self.disk_io.get_path('ifvp', batch_idx=train_batch_idx)
                        ifvp_dict = self.disk_io.load_dict(ifvp_path)
                        train_ifvps = [ifvp_dict[i] if i in ifvp_dict else torch.tensor([])
                                     for i in range(len(self.layer_names))]
                    else:
                        # Get from memory
                        train_ifvps = ifvp_dict[train_batch_idx]

                    # Initialize batch influence
                    batch_influence = None

                    # Compute influence by summing across layers
                    for layer_idx, (train_ifvp, test_grad) in enumerate(zip(train_ifvps, test_grads)):
                        # Skip if either tensor is empty
                        if train_ifvp.numel() == 0 or test_grad.numel() == 0:
                            continue

                        # Move tensors to device using async stream
                        with async_stream():
                            train_ifvp = train_ifvp.to(self.device)
                            test_grad = test_grad.to(self.device)

                            # Compute influence for this layer
                            layer_influence = torch.matmul(train_ifvp, test_grad.t())

                            # Move results back to CPU in the same stream to avoid synchronization
                            cpu_influence = layer_influence.cpu()

                        # Add to batch influence
                        if batch_influence is None:
                            batch_influence = cpu_influence
                        else:
                            batch_influence += cpu_influence

                        # Clean up
                        del train_ifvp, test_grad, layer_influence, cpu_influence
                        torch.cuda.empty_cache()

                    # If we have computed influence for this batch, update the global score
                    if batch_influence is not None:
                        # Update influence scores
                        IF_score[row_st:row_ed, col_st:col_ed] = batch_influence.cpu()

                        # Clean up
                        del batch_influence
                        torch.cuda.empty_cache()

                    # Clean up
                    if self.offload == "disk":
                        del ifvp_dict, train_ifvps

            # Clean up test gradients
            if self.offload == "disk":
                del test_dict, test_grads
            torch.cuda.empty_cache()

        # Return result
        if self.profile and self.profiling_stats:
            return (IF_score, self.profiling_stats)
        else:
            return IF_score

    def _get_ifvp(self, use_cached: bool = True) -> Dict[int, List[torch.Tensor]]:
        """
        Get IFVP - either from memory, disk, or compute new.

        Args:
            use_cached: Whether to use cached IFVP

        Returns:
            Dictionary mapping batch indices to lists of tensors (one tensor per layer)
        """
        # If IFVP are already in memory, return them
        if use_cached and hasattr(self, 'cached_ifvp') and self.cached_ifvp:
            return self.cached_ifvp

        # Otherwise, try to load from disk
        if use_cached and self.offload == "disk":
            # Find all IFVP batch files
            ifvp_files = self.disk_io.find_batch_files('ifvp')

            if ifvp_files:
                print(f"Found {len(ifvp_files)} cached IFVP files on disk.")

                # Just check that files exist, don't actually load them
                # They will be loaded on demand during attribution
                ifvp_dict = {}

                for file_path in ifvp_files:
                    batch_idx = self.disk_io.extract_batch_idx(file_path)
                    # Add a placeholder entry (will be loaded on demand)
                    ifvp_dict[batch_idx] = [torch.tensor([]) for _ in range(len(self.layer_names))]

                # Store in memory (not needed for disk offload, but for consistency)
                self.cached_ifvp = ifvp_dict
                return ifvp_dict

        # If not found, compute them
        print("IFVP not found or cached version not requested. Computing them now...")
        return self.compute_ifvp(save=True)

    def _get_preconditioners(self) -> List[Union[torch.Tensor, str]]:
        """
        Get preconditioners - either from memory or disk.
        Computes them if not already available.

        Returns:
            List of preconditioners for each layer
        """
        # If preconditioners are already in memory, return them
        if hasattr(self, 'preconditioners') and self.preconditioners:
            return self.preconditioners

        # Otherwise, try to load from disk
        if self.offload == "disk":
            preconditioners = [None] * len(self.layer_names)
            preconditioners_found = False

            # Try to load each preconditioner
            for layer_idx in range(len(self.layer_names)):
                precond_path = self.disk_io.get_path('preconditioners', layer_idx=layer_idx)
                if os.path.exists(precond_path):
                    # For disk offload, just store the path - load on demand
                    preconditioners[layer_idx] = precond_path
                    preconditioners_found = True

            # If found, store and return
            if preconditioners_found:
                self.preconditioners = preconditioners
                return preconditioners

        # If not found, compute them
        print("Preconditioners not found. Computing them now...")
        return self.compute_preconditioners(save=True)

    def __del__(self) -> None:
        """
        Clean up resources when the object is garbage collected.
        """
        # Clean up memory
        if hasattr(self, 'hook_manager') and self.hook_manager is not None:
            self.hook_manager.remove_hooks()
            self.hook_manager = None