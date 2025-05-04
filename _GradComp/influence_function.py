from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Literal, List, Optional, Union, Tuple, TypedDict, cast
import os
import time
import threading
import queue
import shutil
import psutil
from dataclasses import dataclass

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    import torch.nn as nn

import torch
from tqdm import tqdm

from .hook import HookManager
from .utils import stable_inverse
from .projector import setup_model_projectors
from .IOManager import IOManager

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

class IFAttributor:
    """
    Influence function calculator using hooks for efficient gradient projection.
    Works with standard PyTorch layers with support for different offloading strategies.
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
        io_threads: int = 32,
        buffer_size_mb: int = 1024,
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
            cache_dir: Directory to save files. Required when offload="disk". Defaults to None.
            io_threads: Number of I/O threads to use. Defaults to 32.
            buffer_size_mb: Size of memory buffer in MB. Defaults to 1024.
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

        # Configure disk storage
        if offload == "disk":
            if cache_dir is None:
                raise ValueError("cache_dir must be provided when offload='disk'")
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)

            # Detect if cache_dir is tmpfs
            is_tmpfs = False
            try:
                cache_stat = os.statvfs(cache_dir)
                mount_points = psutil.disk_partitions(all=True)
                for mount in mount_points:
                    if cache_dir.startswith(mount.mountpoint):
                        is_tmpfs = 'tmpfs' in mount.fstype or mount.mountpoint == '/tmp'
                        break
            except:
                # If we can't determine, assume it's not tmpfs
                is_tmpfs = False

            print(f"Cache directory is {'tmpfs' if is_tmpfs else 'regular disk'}")

            # Initialize I/O manager with appropriate settings
            self.io_manager = IOManager(
                num_threads=io_threads,
                max_queue_size=200,
                high_watermark=0.8,
                low_watermark=0.4,
                use_buffer=True,
                cache_dir=cache_dir,
                buffer_size_mb=buffer_size_mb,
                is_tmpfs=is_tmpfs
            )
            self.io_manager.start()
        else:
            self.cache_dir = None
            self.io_manager = None

        self.full_train_dataloader: Optional[DataLoader] = None
        self.hook_manager: Optional[HookManager] = None
        self.train_gradients: Optional[Dict[int, Dict[int, TensorOrPath]]] = None
        self.projectors: Optional[List[Any]] = None
        self.cached_raw_gradients: Optional[Dict[int, Dict[int, TensorOrPath]]] = None
        self.preconditioners: Optional[Dict[int, TensorOrPath]] = None

        # Store batch information
        self.batch_info = {}

        # Initialize profiling stats
        self.profiling_stats = ProfilingStats() if self.profile else None

    def __del__(self) -> None:
        """Clean up resources when the object is garbage collected."""
        if hasattr(self, 'io_manager') and self.io_manager is not None:
            self.io_manager.stop()

    def _get_file_path(
            self,
            data_type: DataTypeOptions,
            layer_idx: int,
            batch_idx: Optional[int] = None,
            is_test: bool = False
        ) -> str:
        """
        Generate a standardized file path for a specific data type and layer.

        Args:
            data_type: Type of data ('gradients', 'preconditioners', or 'ifvp')
            layer_idx: Index of the layer
            batch_idx: Optional batch index for batch-specific files
            is_test: Whether this is for test data (default: False)

        Returns:
            Full path to the file
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir must be provided when using disk offload")

        # Map data types to subdirectory names
        subdir_map = {
            'gradients': 'grad',
            'preconditioners': 'precond' if self.hessian == "raw" else 'hessian',
            'ifvp': 'ifvp'
        }

        if data_type not in subdir_map:
            raise ValueError(f"Unknown data type: {data_type}")

        subdir = subdir_map[data_type]
        prefix = "test_" if is_test else ""

        # Generate the filename based on data type and context
        if batch_idx is not None:
            # Individual batch-specific filename
            filename = f"{prefix}layer_{layer_idx}_batch_{batch_idx}.pt"
        else:
            # For preconditioners which are aggregated across all batches
            filename = f"{prefix}layer_{layer_idx}.pt"

        # Create the cache subdirectory if it doesn't exist
        cache_subdir = os.path.join(self.cache_dir, subdir)
        os.makedirs(cache_subdir, exist_ok=True)

        return os.path.join(cache_subdir, filename)

    def _save_tensor(
        self,
        tensor: torch.Tensor,
        data_type: DataTypeOptions,
        layer_idx: int,
        batch_idx: Optional[int] = None,
        is_test: bool = False
    ) -> str:
        """
        Save a tensor to disk asynchronously with improved flow control.

        Args:
            tensor: The tensor to save
            data_type: Type of data ('gradients', 'preconditioners', or 'ifvp')
            layer_idx: Index of the layer
            batch_idx: Optional batch index for batch-specific files
            is_test: Whether this is for test data (default: False)

        Returns:
            Path to the saved file
        """
        file_path = self._get_file_path(data_type, layer_idx, batch_idx, is_test)

        if self.offload == "disk" and self.io_manager is not None:
            # Make a detached copy to avoid reference issues
            tensor_copy = tensor.cpu().clone().detach()

            # Queue the save operation for async processing
            self.io_manager.async_write(tensor_copy, file_path)
        else:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Save directly
            torch.save(tensor.cpu(), file_path)

        return file_path

    def _load_tensor(self, file_path: str) -> torch.Tensor:
        """
        Load a tensor from disk synchronously.

        Args:
            file_path: Path to the tensor file

        Returns:
            The loaded tensor
        """
        # Ensure the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find tensor file: {file_path}")

        # Load the tensor
        tensor = torch.load(file_path)
        return tensor

    def _load_tensor_async(self, file_path: str, request_id=None):
        """
        Queue a tensor to be loaded asynchronously.

        Args:
            file_path: Path to the tensor file
            request_id: Optional identifier for the request

        Returns:
            Request ID that can be used to retrieve the tensor later
        """
        if self.offload != "disk" or self.io_manager is None:
            # Fall back to synchronous loading
            tensor = self._load_tensor(file_path)
            return tensor

        return self.io_manager.async_read(file_path, request_id)

    def _wait_for_tensor(self, request_id):
        """
        Wait for an asynchronous tensor load to complete.

        Args:
            request_id: Identifier returned by _load_tensor_async

        Returns:
            The loaded tensor
        """
        if self.offload != "disk" or self.io_manager is None:
            raise ValueError("Cannot wait for async load when not using disk offload")

        return self.io_manager.wait_for_read(request_id)

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
    ) -> Tuple[Dict[int, Dict[int, Union[torch.Tensor, str]]], List[int]]:
        """
        Compute projected gradients for a given dataloader.

        Args:
            dataloader: DataLoader for the data
            batch_range: Tuple of (start_batch, end_batch) to process only a subset of batches
            is_test: Whether this is test data (affects file paths)

        Returns:
            Tuple of (gradients_by_layer_and_batch, batch_sample_counts)
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

        # Initialize storage for gradients (indexed by layer and batch)
        gradients_by_layer = {layer_idx: {} for layer_idx in range(len(self.layer_names))}
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

                # Process and store gradients based on offload strategy
                for layer_idx, grad in enumerate(projected_grads):
                    if grad is None:
                        continue

                    # Detach gradient
                    grad = grad.detach()

                    if self.offload == "none":
                        # Keep on GPU
                        gradients_by_layer[layer_idx][batch_idx] = grad
                    elif self.offload == "cpu":
                        # Move to CPU
                        gradients_by_layer[layer_idx][batch_idx] = grad.cpu()
                        # Free GPU memory
                        del grad
                    elif self.offload == "disk":
                        # Create a detached copy to ensure safe async operations
                        grad_cpu = grad.cpu().clone()

                        # Save batch directly to cache dir and store the path
                        file_path = self._save_tensor(
                            grad_cpu,
                            data_type='gradients',
                            layer_idx=layer_idx,
                            batch_idx=batch_idx,
                            is_test=is_test
                        )
                        gradients_by_layer[layer_idx][batch_idx] = file_path

                        # Free GPU and CPU memory
                        del grad, grad_cpu

            # GPU memory management - ensure we don't run out of memory
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        # Wait for all disk operations to complete if using disk offload
        if self.offload == "disk" and self.io_manager is not None:
            self.io_manager.wait_all()

        return gradients_by_layer, batch_sample_counts

    def cache_gradients(
        self,
        train_dataloader: DataLoader,
        batch_range: Optional[Tuple[int, int]] = None,
        force_cleanup: bool = True
    ) -> Dict[int, Dict[int, TensorOrPath]]:
        """
        Cache raw projected gradients from training data to disk or memory.
        Each batch is stored individually without aggregation for better I/O performance.

        Args:
            train_dataloader: DataLoader for the training data
            batch_range: Optional tuple of (start_batch, end_batch) to process only a subset of batches
            force_cleanup: Whether to force cleanup between batch ranges to prevent resource buildup

        Returns:
            Dictionary mapping layer indices to dictionaries of batch indices to tensors or file paths
        """
        # If batch_range is not provided, process the entire dataset
        if batch_range is None:
            batch_range = (0, len(train_dataloader))

        start_batch, end_batch = batch_range
        batch_msg = f" (processing batches {start_batch} to {end_batch-1})"

        print(f"Caching gradients from training data with offload strategy: {self.offload}{batch_msg}...")
        self.full_train_dataloader = train_dataloader

        # Ensure clean slate for GPU processing
        torch.cuda.empty_cache()

        # Wait for any pending I/O operations
        if self.offload == "disk" and self.io_manager is not None:
            self.io_manager.wait_all()

        # Set up the projectors if not already done
        if self.projectors is None:
            self._setup_projectors(train_dataloader)

        # Compute gradients using common function with batch range
        gradients_by_layer, batch_sample_counts = self._compute_gradients(
            train_dataloader,
            is_test=False,
            batch_range=batch_range
        )

        # Initialize cached_raw_gradients if not already done
        if not hasattr(self, 'cached_raw_gradients') or self.cached_raw_gradients is None:
            self.cached_raw_gradients = {layer_idx: {} for layer_idx in range(len(self.layer_names))}

        # Update with new gradients
        for layer_idx in gradients_by_layer:
            if layer_idx not in self.cached_raw_gradients:
                self.cached_raw_gradients[layer_idx] = {}
            self.cached_raw_gradients[layer_idx].update(gradients_by_layer[layer_idx])

        # Add batch information to the batch_info dictionary
        for batch_idx in range(start_batch, end_batch):
            rel_idx = batch_idx - start_batch
            if rel_idx < len(batch_sample_counts):
                # Calculate offset based on existing entries
                offset = 0
                for existing_idx in sorted(self.batch_info.keys()):
                    if existing_idx < batch_idx:
                        offset += self.batch_info[existing_idx]['sample_count']

                self.batch_info[batch_idx] = {
                    'sample_count': batch_sample_counts[rel_idx],
                    'offset': offset
                }

        # Save batch info to disk when using disk offload
        if self.offload == "disk" and self.cache_dir is not None:
            batch_info_path = os.path.join(self.cache_dir, f"batch_info.pt")
            torch.save({
                'batch_info': self.batch_info,
                'layer_names': self.layer_names
            }, batch_info_path)
            print(f"Updated batch info at {batch_info_path}")

        # Remove hooks after collecting all gradients if we're done
        if force_cleanup and self.hook_manager:
            self.hook_manager.remove_hooks()
            self.hook_manager = None

        # Force cleanup and synchronization to prevent resource buildup
        if force_cleanup:
            torch.cuda.empty_cache()
            if self.offload == "disk" and self.io_manager is not None:
                # Wait for all I/O operations to complete
                self.io_manager.wait_all()

                # Print queue sizes for monitoring
                print(f"I/O Queue sizes - Read: {self.io_manager.read_queue.qsize()}, Write: {self.io_manager.write_queue.qsize()}")

        if self.profile and self.profiling_stats:
            return (gradients_by_layer, self.profiling_stats)
        else:
            return gradients_by_layer

    def _prefetch_gradients(self, layer_idx, batch_indices, max_prefetch=16):
        """
        Prefetch gradients for a layer and multiple batches to improve I/O performance.

        Args:
            layer_idx: Index of the layer
            batch_indices: List of batch indices to prefetch
            max_prefetch: Maximum number of tensors to prefetch at once

        Returns:
            Dictionary mapping batch indices to request IDs
        """
        if self.offload != "disk" or self.io_manager is None:
            return {}

        if not hasattr(self, 'cached_raw_gradients') or not self.cached_raw_gradients:
            return {}

        prefetch_requests = {}
        prefetch_count = 0

        for batch_idx in batch_indices:
            if batch_idx in self.cached_raw_gradients.get(layer_idx, {}):
                file_path = self.cached_raw_gradients[layer_idx][batch_idx]
                if isinstance(file_path, str) and os.path.exists(file_path):
                    request_id = f"grad_{layer_idx}_{batch_idx}"
                    self._load_tensor_async(file_path, request_id)
                    prefetch_requests[batch_idx] = request_id
                    prefetch_count += 1

                    if prefetch_count >= max_prefetch:
                        break

        return prefetch_requests

    def compute_preconditioners(self, damping: Optional[float] = None) -> Dict[int, TensorOrPath]:
        """
        Compute preconditioners (inverse Hessian) from gradients based on the specified Hessian type.
        Uses parallel I/O to load batch gradients efficiently.

        Args:
            damping: Damping factor for Hessian inverse (uses self.damping if None)

        Returns:
            Dictionary mapping layer indices to preconditioners
        """
        print(f"Computing preconditioners with hessian type: {self.hessian}...")

        # Load batch information if not already loaded
        if not self.batch_info:
            if self.offload == "disk" and self.cache_dir is not None:
                batch_info_path = os.path.join(self.cache_dir, "batch_info.pt")
                if os.path.exists(batch_info_path):
                    info_data = torch.load(batch_info_path)
                    self.batch_info = info_data['batch_info']
                    print(f"Loaded batch information from {batch_info_path}")
                else:
                    raise ValueError("No batch information found. Call cache_gradients first.")
            else:
                raise ValueError("No batch information found. Call cache_gradients first.")

        # Use instance damping if not provided
        if damping is None:
            damping = self.damping

        # Calculate total samples across all batches
        total_samples = sum(info['sample_count'] for batch_idx, info in self.batch_info.items())
        print(f"Total samples across all batches: {total_samples}")

        # Calculate Hessian for each layer (one per layer)
        preconditioners = {}

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Process each layer
        for layer_idx, layer_name in tqdm(enumerate(self.layer_names),
                                         desc="Processing layers",
                                         total=len(self.layer_names)):
            # Initialize Hessian accumulator
            hessian_accumulator = None
            sample_count = 0

            # Get sorted batch indices for this layer
            if layer_idx not in self.cached_raw_gradients:
                print(f"Warning: No gradients found for layer {layer_idx}")
                continue

            batch_indices = sorted(self.cached_raw_gradients[layer_idx].keys())

            # Use dynamic chunking to process batches
            # Prefetch up to 16 batches at a time to improve I/O performance
            window_size = 16
            prefetch_window = min(window_size, len(batch_indices))
            prefetch_indices = batch_indices[:prefetch_window]
            prefetch_requests = self._prefetch_gradients(layer_idx, prefetch_indices)
            next_prefetch_idx = prefetch_window

            # Process each batch
            for i, batch_idx in enumerate(batch_indices):
                # Schedule next prefetch if needed
                if next_prefetch_idx < len(batch_indices) and i >= next_prefetch_idx - (window_size // 2):
                    # Prefetch next batch of indices
                    next_window = min(window_size // 2, len(batch_indices) - next_prefetch_idx)
                    next_prefetch = batch_indices[next_prefetch_idx:next_prefetch_idx+next_window]
                    prefetch_requests.update(self._prefetch_gradients(layer_idx, next_prefetch))
                    next_prefetch_idx += next_window

                # Get file path or tensor for this batch
                batch_grad_src = self.cached_raw_gradients[layer_idx].get(batch_idx)
                if batch_grad_src is None:
                    continue

                # Load gradient for this batch
                if isinstance(batch_grad_src, str):
                    # Check if it was prefetched
                    if batch_idx in prefetch_requests:
                        request_id = prefetch_requests.pop(batch_idx)
                        try:
                            batch_grads = self._wait_for_tensor(request_id)
                        except Exception as e:
                            print(f"Error loading prefetched tensor for batch {batch_idx}: {e}")
                            batch_grads = self._load_tensor(batch_grad_src)
                    else:
                        # Load directly
                        batch_grads = self._load_tensor(batch_grad_src)
                else:
                    # Already a tensor
                    batch_grads = batch_grad_src

                # Accumulate batch contribution to Hessian
                # Process in chunks if needed to avoid OOM
                chunk_size = min(1024, batch_grads.shape[0])

                for chunk_start in range(0, batch_grads.shape[0], chunk_size):
                    chunk_end = min(chunk_start + chunk_size, batch_grads.shape[0])

                    # Load chunk to device
                    chunk_grads = batch_grads[chunk_start:chunk_end].to(self.device)

                    # Compute contribution to Hessian
                    chunk_hessian = torch.matmul(chunk_grads.t(), chunk_grads)

                    # Accumulate
                    if hessian_accumulator is None:
                        hessian_accumulator = chunk_hessian
                    else:
                        hessian_accumulator += chunk_hessian

                    # Update sample count
                    sample_count += chunk_grads.shape[0]

                    # Clean up
                    del chunk_grads, chunk_hessian
                    torch.cuda.empty_cache()

                # Clean up batch gradients
                del batch_grads
                torch.cuda.empty_cache()

            # If we have accumulated Hessian, compute preconditioner
            if hessian_accumulator is not None:
                # Normalize by total number of samples
                hessian = hessian_accumulator / sample_count

                # Compute inverse based on Hessian type
                if self.hessian == "raw":
                    precond = stable_inverse(hessian, damping=damping)

                    # Save or store based on offload strategy
                    if self.offload == "disk":
                        # Save directly to cache dir
                        file_path = self._save_tensor(
                            precond.cpu(),
                            data_type='preconditioners',
                            layer_idx=layer_idx
                        )
                        preconditioners[layer_idx] = file_path
                    else:
                        # Store in memory
                        preconditioners[layer_idx] = precond.cpu() if self.cpu_offload else precond

                    # Clean up
                    del precond

                elif self.hessian in ["kfac", "ekfac"]:
                    # Store Hessian itself for KFAC-type preconditioners
                    if self.offload == "disk":
                        # Save directly to cache dir
                        file_path = self._save_tensor(
                            hessian.cpu(),
                            data_type='preconditioners',
                            layer_idx=layer_idx
                        )
                        preconditioners[layer_idx] = file_path
                    else:
                        # Store in memory
                        preconditioners[layer_idx] = hessian.cpu() if self.cpu_offload else hessian

                # Clean up
                del hessian_accumulator, hessian
                torch.cuda.empty_cache()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

        # Wait for all async I/O operations to complete
        if self.offload == "disk" and self.io_manager is not None:
            self.io_manager.wait_all()

        # Store preconditioners
        self.preconditioners = preconditioners

        if self.profile and self.profiling_stats:
            return (preconditioners, self.profiling_stats)
        else:
            return preconditioners

    def compute_ifvp(self) -> Dict[int, Dict[int, TensorOrPath]]:
        """
        Compute inverse-Hessian-vector products (IFVP) from gradients and preconditioners.
        Uses parallel I/O to load batch gradients and apply preconditioners efficiently.

        Returns:
            Dictionary mapping layer indices to dictionaries of batch indices to IFVP tensors or file paths
        """
        print("Computing inverse-Hessian-vector products (IFVP)...")

        # Load batch information if not already loaded
        if not self.batch_info:
            if self.offload == "disk" and self.cache_dir is not None:
                batch_info_path = os.path.join(self.cache_dir, "batch_info.pt")
                if os.path.exists(batch_info_path):
                    info_data = torch.load(batch_info_path)
                    self.batch_info = info_data['batch_info']
                    print(f"Loaded batch information from {batch_info_path}")
                else:
                    raise ValueError("No batch information found. Call cache_gradients first.")
            else:
                raise ValueError("No batch information found. Call cache_gradients first.")

        # Initialize result structure
        ifvp = {layer_idx: {} for layer_idx in range(len(self.layer_names))}

        # Return raw gradients if Hessian type is "none"
        if self.hessian == "none":
            print("Using raw gradients as IFVP since hessian type is 'none'")
            self.train_gradients = self.cached_raw_gradients
            return self.cached_raw_gradients

        # Check if we have preconditioners
        if not hasattr(self, 'preconditioners') or not self.preconditioners:
            print("No preconditioners found. Computing them now...")
            self.compute_preconditioners()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Process each layer
        for layer_idx, layer_name in tqdm(enumerate(self.layer_names),
                                         desc="Processing layers",
                                         total=len(self.layer_names)):
            # Skip if no preconditioner for this layer
            if layer_idx not in self.preconditioners:
                print(f"Warning: No preconditioner found for layer {layer_idx}")
                continue

            # Skip if no gradients for this layer
            if layer_idx not in self.cached_raw_gradients:
                print(f"Warning: No gradients found for layer {layer_idx}")
                continue

            # Load preconditioner for this layer (same for all batches)
            precond_src = self.preconditioners[layer_idx]
            if isinstance(precond_src, str):
                precond = self._load_tensor(precond_src).to(self.device)
            else:
                precond = precond_src.to(self.device)

            # Get sorted batch indices for this layer
            batch_indices = sorted(self.cached_raw_gradients[layer_idx].keys())

            # Use dynamic chunking to process batches
            # Prefetch up to 16 batches at a time to improve I/O performance
            window_size = 16
            prefetch_window = min(window_size, len(batch_indices))
            prefetch_indices = batch_indices[:prefetch_window]
            prefetch_requests = self._prefetch_gradients(layer_idx, prefetch_indices)
            next_prefetch_idx = prefetch_window

            # Process each batch
            for i, batch_idx in enumerate(batch_indices):
                # Schedule next prefetch if needed
                if next_prefetch_idx < len(batch_indices) and i >= next_prefetch_idx - (window_size // 2):
                    # Prefetch next batch of indices
                    next_window = min(window_size // 2, len(batch_indices) - next_prefetch_idx)
                    next_prefetch = batch_indices[next_prefetch_idx:next_prefetch_idx+next_window]
                    prefetch_requests.update(self._prefetch_gradients(layer_idx, next_prefetch))
                    next_prefetch_idx += next_window

                # Get file path or tensor for this batch
                batch_grad_src = self.cached_raw_gradients[layer_idx].get(batch_idx)
                if batch_grad_src is None:
                    continue

                # Load gradient for this batch
                if isinstance(batch_grad_src, str):
                    # Check if it was prefetched
                    if batch_idx in prefetch_requests:
                        request_id = prefetch_requests.pop(batch_idx)
                        try:
                            batch_grads = self._wait_for_tensor(request_id)
                        except Exception as e:
                            print(f"Error loading prefetched tensor for batch {batch_idx}: {e}")
                            batch_grads = self._load_tensor(batch_grad_src)
                    else:
                        # Load directly
                        batch_grads = self._load_tensor(batch_grad_src)
                else:
                    # Already a tensor
                    batch_grads = batch_grad_src

                # Compute IFVP for this batch
                # Process in smaller chunks if needed to avoid OOM
                batch_size = min(1024, batch_grads.shape[0])
                result_tensor = torch.zeros((batch_grads.shape[0], precond.shape[0]),
                                    dtype=precond.dtype)

                for chunk_start in range(0, batch_grads.shape[0], batch_size):
                    chunk_end = min(chunk_start + batch_size, batch_grads.shape[0])
                    grads_chunk = batch_grads[chunk_start:chunk_end].to(self.device)

                    # Compute IFVP for this chunk
                    result_chunk = torch.matmul(precond, grads_chunk.t()).t()

                    # Store result
                    result_tensor[chunk_start:chunk_end] = result_chunk.cpu()

                    # Clean up
                    del grads_chunk, result_chunk
                    torch.cuda.empty_cache()

                # Save result based on offload strategy
                if self.offload == "disk":
                    # Save directly to cache dir
                    file_path = self._save_tensor(
                        result_tensor,
                        data_type='ifvp',
                        layer_idx=layer_idx,
                        batch_idx=batch_idx
                    )
                    ifvp[layer_idx][batch_idx] = file_path
                else:
                    # Store in memory
                    ifvp[layer_idx][batch_idx] = result_tensor.cpu() if self.cpu_offload else result_tensor

                # Clean up
                del batch_grads, result_tensor
                torch.cuda.empty_cache()

            # Clean up preconditioner
            del precond
            torch.cuda.empty_cache()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

        # Wait for all async I/O operations to complete
        if self.offload == "disk" and self.io_manager is not None:
            self.io_manager.wait_all()

        # Store IFVP results
        self.train_gradients = ifvp

        if self.profile and self.profiling_stats:
            return (ifvp, self.profiling_stats)
        else:
            return ifvp

    def attribute(
        self,
        test_dataloader: DataLoader,
        train_dataloader: Optional[DataLoader] = None,
        use_cached_ifvp: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ProfilingStats]]:
        """
        Attribute influence of training examples on test examples.
        Uses parallel I/O to load data efficiently and processes in small chunks to manage memory.

        Args:
            test_dataloader: DataLoader for test data
            train_dataloader: Optional DataLoader for training data if not cached
            use_cached_ifvp: Whether to use cached IFVP (True) or recompute from cached gradients (False)

        Returns:
            Tensor of influence scores (and profiling stats if profile=True)
        """
        # Load batch information if not already loaded
        if not self.batch_info:
            if self.offload == "disk" and self.cache_dir is not None:
                batch_info_path = os.path.join(self.cache_dir, "batch_info.pt")
                if os.path.exists(batch_info_path):
                    info_data = torch.load(batch_info_path)
                    self.batch_info = info_data['batch_info']
                    print(f"Loaded batch information from {batch_info_path}")
                else:
                    if train_dataloader is None:
                        raise ValueError("No batch information found and no training dataloader provided.")
            elif train_dataloader is None:
                raise ValueError("No batch information found and no training dataloader provided.")

        # Validate input
        if train_dataloader is None and self.full_train_dataloader is None and not hasattr(self, 'cached_raw_gradients'):
            raise ValueError("No training data provided or cached.")

        # Get cached IFVP or calculate new ones
        ifvp_train = None

        if use_cached_ifvp and hasattr(self, 'train_gradients') and self.train_gradients:
            # Use in-memory cached IFVP
            ifvp_train = self.train_gradients
            print("Using cached IFVP for attribution.")
        else:
            # Check for cached preconditioners and gradients
            preconditioners_found = hasattr(self, 'preconditioners') and self.preconditioners
            raw_gradients_found = hasattr(self, 'cached_raw_gradients') and self.cached_raw_gradients

            # Load from disk if using disk offload
            if self.offload == "disk" and self.cache_dir is not None:
                # Check for preconditioner files
                if not preconditioners_found:
                    preconditioners = {}
                    for layer_idx in range(len(self.layer_names)):
                        precond_path = self._get_file_path('preconditioners', layer_idx)
                        if os.path.exists(precond_path):
                            preconditioners[layer_idx] = precond_path
                            preconditioners_found = True

                    if preconditioners_found:
                        print("Found preconditioners on disk.")
                        self.preconditioners = preconditioners

                # Check for gradient files
                if not raw_gradients_found:
                    # Load batch info to get batch indices
                    if not self.batch_info:
                        batch_info_path = os.path.join(self.cache_dir, "batch_info.pt")
                        if os.path.exists(batch_info_path):
                            info_data = torch.load(batch_info_path)
                            self.batch_info = info_data['batch_info']

                    if self.batch_info:
                        gradients = {layer_idx: {} for layer_idx in range(len(self.layer_names))}
                        for layer_idx in range(len(self.layer_names)):
                            for batch_idx in self.batch_info.keys():
                                grad_path = self._get_file_path('gradients', layer_idx, batch_idx=batch_idx)
                                if os.path.exists(grad_path):
                                    gradients[layer_idx][batch_idx] = grad_path
                                    raw_gradients_found = True

                        if raw_gradients_found:
                            print("Found raw gradients on disk.")
                            self.cached_raw_gradients = gradients

            # Compute IFVP if we have both gradients and preconditioners (or hessian is "none")
            if (raw_gradients_found and (preconditioners_found or self.hessian == "none")):
                print("Computing IFVP from cached gradients and preconditioners...")
                ifvp_train = self.compute_ifvp()
            elif train_dataloader is not None:
                # New train data, cache everything
                print("No complete cached data found. Caching gradients from provided dataloader...")
                self.full_train_dataloader = train_dataloader
                self.cache_gradients(train_dataloader)

                if self.hessian != "none" and not preconditioners_found:
                    print("Computing preconditioners...")
                    self.compute_preconditioners()

                print("Computing IFVP...")
                ifvp_train = self.compute_ifvp()
            else:
                raise ValueError("No training data provided and insufficient cached data found.")

        # Set up the projectors if not already done
        if self.projectors is None:
            self._setup_projectors(test_dataloader)

        # Compute test gradients
        test_gradients, test_batch_sample_counts = self._compute_gradients(
            test_dataloader,
            is_test=True,
            batch_range=(0, len(test_dataloader))
        )

        # Calculate total number of test samples
        num_test = sum(test_batch_sample_counts)

        # Calculate total number of training samples
        num_train = sum(info['sample_count'] for batch_idx, info in self.batch_info.items())

        if num_train == 0:
            raise ValueError("Cannot determine number of training samples")

        # Initialize influence scores in memory
        IF_score = torch.zeros(num_train, num_test)

        # Create mapping from test batch index to column range in the result tensor
        test_batch_indices = {}
        col_offset = 0
        for i, batch_idx in enumerate(sorted(test_gradients[0].keys())):
            batch_size = test_batch_sample_counts[i]
            test_batch_indices[batch_idx] = (col_offset, col_offset + batch_size)
            col_offset += batch_size

        # Create mapping from train batch index to row range in the result tensor
        train_batch_indices = {}
        for batch_idx, info in self.batch_info.items():
            row_start = info['offset']
            row_end = row_start + info['sample_count']
            train_batch_indices[batch_idx] = (row_start, row_end)

        # Process each layer
        for layer_idx, layer_name in tqdm(enumerate(self.layer_names),
                                         desc="Processing layers",
                                         total=len(self.layer_names)):
            # Skip if no test gradients or train gradients for this layer
            if layer_idx not in test_gradients or layer_idx not in ifvp_train:
                continue

            # Get test batch indices for this layer
            test_batch_ids = sorted(test_gradients[layer_idx].keys())

            # Get train batch indices for this layer
            train_batch_ids = sorted(ifvp_train[layer_idx].keys())

            # Use dynamic prefetching for test batches
            prefetch_test_window = min(8, len(test_batch_ids))
            test_prefetch_requests = {}

            # Prefetch initial test batches
            for test_idx in range(prefetch_test_window):
                if test_idx < len(test_batch_ids):
                    test_batch_idx = test_batch_ids[test_idx]
                    test_grad_src = test_gradients[layer_idx][test_batch_idx]
                    if isinstance(test_grad_src, str):
                        request_id = f"test_{layer_idx}_{test_batch_idx}"
                        self._load_tensor_async(test_grad_src, request_id)
                        test_prefetch_requests[test_batch_idx] = request_id

            # Process each train batch
            for train_batch_idx in tqdm(train_batch_ids, desc=f"Processing train batches for layer {layer_idx}"):
                # Skip if no mapping for this batch
                if train_batch_idx not in train_batch_indices:
                    continue

                # Get row indices for this batch
                row_start, row_end = train_batch_indices[train_batch_idx]

                # Get train gradients for this batch
                train_grad_src = ifvp_train[layer_idx][train_batch_idx]
                if isinstance(train_grad_src, str):
                    train_grads = self._load_tensor(train_grad_src)
                else:
                    train_grads = train_grad_src

                # Process each test batch with dynamic prefetching
                for test_idx, test_batch_idx in enumerate(test_batch_ids):
                    # Schedule next test prefetch if needed
                    next_test_idx = test_idx + prefetch_test_window
                    if next_test_idx < len(test_batch_ids):
                        next_test_batch_idx = test_batch_ids[next_test_idx]
                        test_grad_src = test_gradients[layer_idx][next_test_batch_idx]
                        if isinstance(test_grad_src, str) and next_test_batch_idx not in test_prefetch_requests:
                            request_id = f"test_{layer_idx}_{next_test_batch_idx}"
                            self._load_tensor_async(test_grad_src, request_id)
                            test_prefetch_requests[next_test_batch_idx] = request_id

                    # Skip if no mapping for this test batch
                    if test_batch_idx not in test_batch_indices:
                        continue

                    # Get column indices for this test batch
                    col_start, col_end = test_batch_indices[test_batch_idx]

                    # Get test gradients for this batch
                    test_grad_src = test_gradients[layer_idx][test_batch_idx]
                    if isinstance(test_grad_src, str):
                        # Check if it was prefetched
                        if test_batch_idx in test_prefetch_requests:
                            request_id = test_prefetch_requests.pop(test_batch_idx)
                            try:
                                test_grads = self._wait_for_tensor(request_id)
                            except Exception as e:
                                print(f"Error loading prefetched test tensor for batch {test_batch_idx}: {e}")
                                test_grads = self._load_tensor(test_grad_src)
                        else:
                            # Load directly
                            test_grads = self._load_tensor(test_grad_src)
                    else:
                        test_grads = test_grad_src

                    # Compute influence scores in chunks to manage memory
                    train_chunk_size = min(1024, train_grads.shape[0])
                    for train_chunk_start in range(0, train_grads.shape[0], train_chunk_size):
                        train_chunk_end = min(train_chunk_start + train_chunk_size, train_grads.shape[0])
                        train_chunk = train_grads[train_chunk_start:train_chunk_end].to(self.device)

                        # Map local chunk indices to global row indices
                        global_row_start = row_start + train_chunk_start
                        global_row_end = row_start + train_chunk_end

                        # Process test data in chunks as well if needed
                        test_chunk_size = min(1024, test_grads.shape[0])
                        for test_chunk_start in range(0, test_grads.shape[0], test_chunk_size):
                            test_chunk_end = min(test_chunk_start + test_chunk_size, test_grads.shape[0])
                            test_chunk = test_grads[test_chunk_start:test_chunk_end].to(self.device)

                            # Map local chunk indices to global column indices
                            global_col_start = col_start + test_chunk_start
                            global_col_end = col_start + test_chunk_end

                            # Compute influence for this chunk pair
                            result = torch.matmul(train_chunk, test_chunk.t())

                            # Update influence scores
                            IF_score[global_row_start:global_row_end, global_col_start:global_col_end] += result.cpu()

                            # Clean up
                            del test_chunk, result
                            torch.cuda.empty_cache()

                        # Clean up
                        del train_chunk
                        torch.cuda.empty_cache()

                    # Clean up test gradients
                    del test_grads
                    torch.cuda.empty_cache()

                # Clean up train gradients
                del train_grads
                torch.cuda.empty_cache()

        # Clean up test gradients
        if self.hook_manager:
            self.hook_manager.remove_hooks()
            self.hook_manager = None

        # Wait for any remaining I/O operations
        if self.offload == "disk" and self.io_manager is not None:
            self.io_manager.wait_all()

        # Return result
        if self.profile and self.profiling_stats:
            return (IF_score, self.profiling_stats)
        else:
            return IF_score