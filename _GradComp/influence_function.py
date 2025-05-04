from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Literal, List, Optional, Union, Tuple, TypedDict, cast
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
from dataclasses import dataclass
import shutil
from abc import ABC, abstractmethod

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
    disk_io: float = 0.0  # Fixed typo from 'flaot' to 'float'


class IOManager(ABC):
    """Base class for IO operations with different offload strategies."""

    @abstractmethod
    def save_tensor(self, tensor: torch.Tensor, path: str) -> str:
        """Save a tensor to the specified path."""
        pass

    @abstractmethod
    def load_tensor(self, path: str) -> torch.Tensor:
        """Load a tensor from the specified path."""
        pass


class MemoryIOManager(IOManager):
    """IO Manager for in-memory operations."""

    def __init__(self, device: str, cpu_offload: bool = False):
        self.device = device
        self.cpu_offload = cpu_offload
        # Dictionary to store tensors in memory
        self.tensor_cache: Dict[str, torch.Tensor] = {}

    def save_tensor(self, tensor: torch.Tensor, path: str) -> str:
        """Store tensor in memory cache."""
        if self.cpu_offload:
            tensor = tensor.cpu()
        self.tensor_cache[path] = tensor
        return path

    def load_tensor(self, path: str) -> torch.Tensor:
        """Load tensor from memory cache."""
        if path not in self.tensor_cache:
            raise FileNotFoundError(f"Tensor not found in memory cache: {path}")

        tensor = self.tensor_cache[path]
        if not self.cpu_offload:
            tensor = tensor.to(self.device)
        return tensor


class DiskIOManager(IOManager):
    """IO Manager for disk-based operations with parallel I/O."""

    def __init__(self, cache_dir: str, device: str, max_workers: int = 32,
                 max_queue_size: int = 100, max_cache_size: int = 20):
        self.cache_dir = cache_dir
        self.device = device
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.max_cache_size = max_cache_size

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Create in-memory LRU cache
        self.tensor_cache: Dict[str, torch.Tensor] = {}
        self.cache_order: List[str] = []

        # Set up thread pool for I/O operations
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Queue for prefetching
        self.prefetch_queue = queue.Queue(maxsize=self.max_queue_size)

        # Start prefetch worker thread
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()

        # Track futures for async operations
        self.pending_futures = set()

    def save_tensor(self, tensor: torch.Tensor, path: str) -> str:
        """Save tensor to disk asynchronously."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Move tensor to CPU for saving
        tensor_cpu = tensor.cpu()

        # Submit save task to executor
        future = self.executor.submit(torch.save, tensor_cpu, path)
        self.pending_futures.add(future)
        future.add_done_callback(lambda f: self.pending_futures.remove(f))

        # Add to in-memory cache
        self._add_to_cache(path, tensor_cpu)

        return path

    def load_tensor(self, path: str) -> torch.Tensor:
        """Load tensor from disk or cache."""
        # Check if in memory cache first
        if path in self.tensor_cache:
            # Update position in cache order
            self.cache_order.remove(path)
            self.cache_order.append(path)
            return self.tensor_cache[path].to(self.device)

        # Load from disk
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tensor file not found: {path}")

        tensor = torch.load(path)

        # Add to cache
        self._add_to_cache(path, tensor)

        return tensor.to(self.device)

    def queue_prefetch(self, paths: List[str]) -> None:
        """Queue paths for prefetching."""
        for path in paths:
            try:
                self.prefetch_queue.put(path, block=False)
            except queue.Full:
                # Queue is full, skip prefetching this path
                pass

    def _prefetch_worker(self) -> None:
        """Worker thread that prefetches tensors in background."""
        while True:
            try:
                path = self.prefetch_queue.get()

                # Skip if already in cache
                if path in self.tensor_cache:
                    self.prefetch_queue.task_done()
                    continue

                # Check if file exists
                if not os.path.exists(path):
                    self.prefetch_queue.task_done()
                    continue

                # Load tensor and add to cache
                tensor = torch.load(path)
                self._add_to_cache(path, tensor)

                self.prefetch_queue.task_done()
            except Exception as e:
                print(f"Error in prefetch worker: {e}")
                self.prefetch_queue.task_done()

    def _add_to_cache(self, path: str, tensor: torch.Tensor) -> None:
        """Add tensor to in-memory cache with LRU eviction."""
        # Add to cache
        self.tensor_cache[path] = tensor

        # Update cache order
        if path in self.cache_order:
            self.cache_order.remove(path)
        self.cache_order.append(path)

        # Evict if cache is full
        while len(self.cache_order) > self.max_cache_size:
            oldest_path = self.cache_order.pop(0)
            del self.tensor_cache[oldest_path]

    def wait_for_pending(self) -> None:
        """Wait for all pending async operations to complete."""
        for future in list(self.pending_futures):
            future.result()


class BaseAttributor(ABC):
    """
    Base class for influence attribution methods.
    Defines common interface and utility methods for attribution algorithms.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: Union[str, List[str]],
        device: str = 'cpu',
        profile: bool = False,
    ) -> None:
        """
        Initialize base attributor.

        Args:
            model: PyTorch model
            layer_names: Names of layers to attribute
            device: Device to run the model on
            profile: Whether to profile performance
        """
        self.model = model
        self.model.to(device)
        self.model.eval()

        # Ensure layer_names is a list
        self.layer_names = [layer_names] if isinstance(layer_names, str) else layer_names

        self.device = device
        self.profile = profile

        # Hooks and projectors
        self.hook_manager: Optional[HookManager] = None
        self.projectors: Optional[List[Any]] = None

        # Profiling stats
        self.profiling_stats = ProfilingStats() if self.profile else None

    @abstractmethod
    def _setup_projectors(self, train_dataloader: DataLoader) -> None:
        """Set up projectors for the model layers."""
        pass

    @abstractmethod
    def cache_gradients(
        self,
        train_dataloader: DataLoader,
        batch_range: Optional[Tuple[int, int]] = None,
        worker_id: Optional[int] = None
    ) -> List[Dict[int, TensorOrPath]]:
        """Cache gradients from training data."""
        pass

    @abstractmethod
    def compute_preconditioners(self, damping: Optional[float] = None) -> List[TensorOrPath]:
        """Compute preconditioners for the model."""
        pass

    @abstractmethod
    def compute_ifvp(self) -> List[Dict[int, TensorOrPath]]:
        """Compute inverse-Hessian-vector products."""
        pass

    @abstractmethod
    def attribute(
        self,
        test_dataloader: DataLoader,
        train_dataloader: Optional[DataLoader] = None,
        use_cached_ifvp: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ProfilingStats]]:
        """Attribute influence of training examples on test examples."""
        pass


class IFAttributor(BaseAttributor):
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
            cache_dir: Directory to save final IFVP files. Only used when offload="disk". Defaults to None.
        """
        # Initialize base class
        super().__init__(model, layer_names, device, profile)

        self.setting = setting
        self.hessian = hessian
        self.damping = damping
        self.projector_kwargs = projector_kwargs or {}
        self.offload = offload
        self.cpu_offload = offload in ["cpu", "disk"]

        # Set up cache directory
        self.cache_dir = cache_dir
        if self.offload == "disk" and self.cache_dir is not None:
            # Create cache directory structure
            self._setup_cache_dirs()

        # Initialize IO manager based on offload strategy
        self.io_manager = self._initialize_io_manager()

        self.full_train_dataloader: Optional[DataLoader] = None
        self.train_gradients: Optional[List[Dict[int, TensorOrPath]]] = None
        self.cached_raw_gradients: Optional[List[Dict[int, TensorOrPath]]] = None
        self.preconditioners: Optional[List[TensorOrPath]] = None

        # Batch information
        self.batch_info: Dict[int, Dict[str, Any]] = {}

    def _initialize_io_manager(self) -> IOManager:
        """Initialize the appropriate IO manager based on offload strategy."""
        if self.offload == "disk":
            if self.cache_dir is None:
                raise ValueError("cache_dir must be provided when offload='disk'")
            return DiskIOManager(self.cache_dir, self.device)
        else:
            return MemoryIOManager(self.device, self.cpu_offload)

    def _setup_cache_dirs(self) -> None:
        """Set up cache directory structure."""
        if self.cache_dir is None:
            return

        # Create main directories
        os.makedirs(self.cache_dir, exist_ok=True)

        # Create data type directories
        os.makedirs(os.path.join(self.cache_dir, "grad"), exist_ok=True)

        precond_dir = "precond" if self.hessian == "raw" else "hessian"
        os.makedirs(os.path.join(self.cache_dir, precond_dir), exist_ok=True)

        os.makedirs(os.path.join(self.cache_dir, "ifvp"), exist_ok=True)

        # Create layer directories for gradients and IFVP
        for layer_idx in range(len(self.layer_names)):
            os.makedirs(os.path.join(self.cache_dir, "grad", f"layer_{layer_idx}"), exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, "ifvp", f"layer_{layer_idx}"), exist_ok=True)

    def __del__(self) -> None:
        """Clean up resources when the object is garbage collected."""
        pass  # No manual cleanup needed with our new approach

    def _get_file_path(
            self,
            data_type: DataTypeOptions,
            layer_idx: int,
            batch_idx: Optional[int] = None,
            worker_id: Optional[int] = None
        ) -> str:
        """
        Generate a standardized file path for a specific data type and layer.

        Args:
            data_type: Type of data ('gradients', 'preconditioners', or 'ifvp')
            layer_idx: Index of the layer
            batch_idx: Optional batch index for individual batch files
            worker_id: Optional identifier for the worker

        Returns:
            Full path to the file
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir is required for file operations")

        # Map data types to subdirectory names
        subdir_map = {
            'gradients': 'grad',
            'preconditioners': 'precond' if self.hessian == "raw" else 'hessian',
            'ifvp': 'ifvp'
        }

        if data_type not in subdir_map:
            raise ValueError(f"Unknown data type: {data_type}")

        subdir = subdir_map[data_type]

        # Determine filename
        if data_type == 'preconditioners':
            # Preconditioners are per layer (no batches)
            return os.path.join(self.cache_dir, subdir, f"layer_{layer_idx}.pt")
        else:
            # Gradients and IFVP have per-batch files
            layer_dir = os.path.join(self.cache_dir, subdir, f"layer_{layer_idx}")

            if batch_idx is not None:
                # Individual batch file
                if worker_id is not None:
                    return os.path.join(layer_dir, f"worker_{worker_id}_batch_{batch_idx}.pt")
                else:
                    return os.path.join(layer_dir, f"batch_{batch_idx}.pt")
            elif worker_id is not None:
                # Combined file for a worker
                return os.path.join(layer_dir, f"worker_{worker_id}.pt")
            else:
                raise ValueError("Either batch_idx or worker_id must be provided for gradients/ifvp")

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
    ) -> Tuple[List[List[Union[torch.Tensor, str]]], List[int]]:
        """
        Compute projected gradients for a given dataloader.

        Args:
            dataloader: DataLoader for the data
            batch_range: Tuple of (start_batch, end_batch) to process only a subset of batches
            is_test: Whether this is test data (affects file paths)

        Returns:
            Tuple of (per_layer_gradients, batch_sample_counts)
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

        # Initialize storage for gradients
        per_layer_gradients = [[] for _ in self.layer_names]
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
                        per_layer_gradients[layer_idx].append(grad)
                    elif self.offload == "cpu":
                        # Move to CPU
                        per_layer_gradients[layer_idx].append(grad.cpu())
                        # Free GPU memory
                        del grad
                    elif self.offload == "disk":
                        # For disk mode, directly save each batch to the appropriate location
                        if is_test:
                            # For test data, we simply keep as a list for now
                            per_layer_gradients[layer_idx].append(grad.cpu())
                        else:
                            # For training data, save directly to disk
                            worker_id = batch_range[2] if len(batch_range) > 2 else None
                            file_path = self._get_file_path(
                                'gradients',
                                layer_idx,
                                batch_idx=batch_idx,
                                worker_id=worker_id
                            )

                            # Log disk I/O time if profiling
                            if self.profile and self.profiling_stats:
                                start_io = time.time()

                            # Save tensor
                            self.io_manager.save_tensor(grad.cpu(), file_path)

                            if self.profile and self.profiling_stats:
                                self.profiling_stats.disk_io += time.time() - start_io

                            # Store path in output
                            per_layer_gradients[layer_idx].append(file_path)

                        # Free GPU memory
                        del grad

            # GPU memory management - ensure we don't run out of memory
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        # Wait for all pending IO operations
        if self.offload == "disk":
            if isinstance(self.io_manager, DiskIOManager):
                self.io_manager.wait_for_pending()

        return per_layer_gradients, batch_sample_counts

    def cache_gradients(
        self,
        train_dataloader: DataLoader,
        batch_range: Optional[Tuple[int, int]] = None,
        worker_id: Optional[int] = None
    ) -> List[Dict[int, TensorOrPath]]:
        """
        Cache raw projected gradients from training data to disk or memory.
        Each batch is stored individually without aggregation.

        Args:
            train_dataloader: DataLoader for the training data
            batch_range: Optional tuple of (start_batch, end_batch) to process only a subset of batches
            worker_id: Optional identifier for the batch group when using parallel processing

        Returns:
            List of dictionaries mapping batch indices to tensor or filename containing gradients for each layer
        """
        # If batch_range and worker_id are not provided, process the entire dataset
        if batch_range is None:
            batch_range = (0, len(train_dataloader))

        if worker_id is None:
            worker_id = 0

        extended_batch_range = (batch_range[0], batch_range[1], worker_id)
        print(f"Caching gradients with worker_id={worker_id}, batch range: {batch_range}")

        # Store dataloader for future use
        self.full_train_dataloader = train_dataloader

        # Set up the projectors if not already done
        if self.projectors is None:
            self._setup_projectors(train_dataloader)

        # Compute gradients using common function with batch range
        per_layer_gradients, batch_sample_counts = self._compute_gradients(
            train_dataloader,
            is_test=False,
            batch_range=extended_batch_range
        )

        # Process collected gradients
        gradients: List[Dict[int, TensorOrPath]] = [{} for _ in self.layer_names]

        # Store each batch individually
        for layer_idx, _ in enumerate(self.layer_names):
            if not per_layer_gradients[layer_idx]:
                continue

            if self.offload == "disk":
                # For disk mode, paths are already stored correctly in per_layer_gradients
                # We just need to structure them correctly in the output
                for batch_idx, file_path in enumerate(per_layer_gradients[layer_idx]):
                    actual_batch_idx = batch_range[0] + batch_idx
                    gradients[layer_idx][actual_batch_idx] = file_path
            else:
                # For memory modes, we need to save each batch separately
                for batch_idx, grad in enumerate(per_layer_gradients[layer_idx]):
                    actual_batch_idx = batch_range[0] + batch_idx
                    gradients[layer_idx][actual_batch_idx] = grad

        print(f"Cached gradients for {len(self.layer_names)} modules")

        # Store the raw gradients
        self.cached_raw_gradients = gradients

        # Store batch information
        self.batch_info[worker_id] = {
            'batch_range': batch_range,
            'sample_counts': batch_sample_counts,
            'total_samples': sum(batch_sample_counts)
        }

        # Save batch info to disk when using disk offload
        if self.offload == "disk" and self.cache_dir is not None:
            batch_info_path = os.path.join(self.cache_dir, f"batch_info_{worker_id}.pt")
            torch.save({
                'batch_info': {worker_id: self.batch_info[worker_id]},
                'layer_names': self.layer_names
            }, batch_info_path)
            print(f"Saved batch info to {batch_info_path}")

        # Remove hooks after collecting all gradients
        if self.hook_manager:
            self.hook_manager.remove_hooks()
            self.hook_manager = None

        if self.profile and self.profiling_stats:
            return (gradients, self.profiling_stats)
        else:
            return gradients

    def compute_preconditioners(self, damping: Optional[float] = None) -> List[TensorOrPath]:
        """
        Compute preconditioners (inverse Hessian) from gradients based on the specified Hessian type.
        Accumulates Hessian contributions from all batches to compute a single preconditioner per layer.

        Args:
            damping: Damping factor for Hessian inverse (uses self.damping if None)

        Returns:
            List of preconditioners for each layer (one preconditioner per layer)
        """
        print(f"Computing preconditioners with hessian type: {self.hessian}...")

        # Load batch information if not already loaded
        if not self.batch_info:
            self._load_batch_info()

        # Use instance damping if not provided
        if damping is None:
            damping = self.damping

        # Calculate total samples across all batches
        total_samples = sum(info['total_samples'] for worker_id, info in self.batch_info.items())
        print(f"Total samples across all batches: {total_samples}")

        # Calculate Hessian for each layer (one per layer, not per batch)
        preconditioners: List[Optional[TensorOrPath]] = [None] * len(self.layer_names)

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Process each layer
        for layer_idx, _ in tqdm(enumerate(self.layer_names), desc="Processing layers", total=len(self.layer_names)):
            # Initialize Hessian accumulator
            hessian_accumulator = None
            sample_count = 0

            # Process all batches for this layer across all workers
            for worker_id in sorted(self.batch_info.keys()):
                batch_range = self.batch_info[worker_id]['batch_range']
                start_batch, end_batch = batch_range

                # Queue batches to prefetch if using disk IO
                if self.offload == "disk" and isinstance(self.io_manager, DiskIOManager):
                    prefetch_paths = []
                    for batch_idx in range(start_batch, end_batch):
                        grad_path = self._get_file_path(
                            'gradients',
                            layer_idx,
                            batch_idx=batch_idx,
                            worker_id=worker_id
                        )
                        prefetch_paths.append(grad_path)

                    self.io_manager.queue_prefetch(prefetch_paths)

                # Process each batch for this worker
                for batch_idx in range(start_batch, end_batch):
                    # Get gradient path for this batch
                    grad_path = None

                    if self.offload == "disk":
                        grad_path = self._get_file_path(
                            'gradients',
                            layer_idx,
                            batch_idx=batch_idx,
                            worker_id=worker_id
                        )

                        # Skip if file doesn't exist
                        if not os.path.exists(grad_path):
                            continue
                    elif batch_idx in self.cached_raw_gradients[layer_idx]:
                        grad_path = batch_idx  # Use batch_idx as key for memory modes
                    else:
                        continue

                    # Load gradient
                    if self.profile and self.profiling_stats:
                        start_io = time.time()

                    if self.offload == "disk":
                        batch_grads = self.io_manager.load_tensor(grad_path)
                    else:
                        batch_grads = self.cached_raw_gradients[layer_idx][grad_path]
                        if self.offload == "cpu":
                            batch_grads = batch_grads.to(self.device)

                    if self.profile and self.profiling_stats:
                        self.profiling_stats.disk_io += time.time() - start_io

                    # Process in chunks if needed to avoid OOM
                    chunk_size = min(1024, batch_grads.shape[0])

                    for chunk_start in range(0, batch_grads.shape[0], chunk_size):
                        chunk_end = min(chunk_start + chunk_size, batch_grads.shape[0])

                        # Get chunk
                        chunk_grads = batch_grads[chunk_start:chunk_end]

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
                        file_path = self._get_file_path('preconditioners', layer_idx)

                        if self.profile and self.profiling_stats:
                            start_io = time.time()

                        self.io_manager.save_tensor(precond.cpu(), file_path)

                        if self.profile and self.profiling_stats:
                            self.profiling_stats.disk_io += time.time() - start_io

                        preconditioners[layer_idx] = file_path
                    else:
                        preconditioners[layer_idx] = precond.cpu() if self.cpu_offload else precond

                    # Clean up
                    del precond

                elif self.hessian in ["kfac", "ekfac"]:
                    # Store Hessian itself for KFAC-type preconditioners
                    if self.offload == "disk":
                        file_path = self._get_file_path('preconditioners', layer_idx)

                        if self.profile and self.profiling_stats:
                            start_io = time.time()

                        self.io_manager.save_tensor(hessian.cpu(), file_path)

                        if self.profile and self.profiling_stats:
                            self.profiling_stats.disk_io += time.time() - start_io

                        preconditioners[layer_idx] = file_path
                    else:
                        preconditioners[layer_idx] = hessian.cpu() if self.cpu_offload else hessian

                # Clean up
                del hessian_accumulator, hessian
                torch.cuda.empty_cache()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

        # Store preconditioners
        self.preconditioners = preconditioners

        if self.profile and self.profiling_stats:
            return (preconditioners, self.profiling_stats)
        else:
            return preconditioners

    def _load_batch_info(self) -> None:
        """Load batch information from disk if not already in memory."""
        if self.batch_info:
            return

        if self.offload == "disk" and self.cache_dir is not None:
            # Look for batch info files
            try:
                batch_info_files = [f for f in os.listdir(self.cache_dir)
                                if f.startswith("batch_info_") and f.endswith(".pt")]
            except FileNotFoundError:
                print(f"Warning: Cache directory {self.cache_dir} not found")
                batch_info_files = []

            if batch_info_files:
                # Load batch info from files
                self.batch_info = {}
                for info_file in batch_info_files:
                    file_path = os.path.join(self.cache_dir, info_file)
                    info_data = torch.load(file_path)
                    # Update batch info
                    self.batch_info.update(info_data['batch_info'])

                print(f"Loaded batch information for {len(self.batch_info)} batches from disk")
            else:
                raise ValueError("No batch information found. Call cache_gradients first.")
        else:
            raise ValueError("No batch information found. Call cache_gradients first.")

    def compute_ifvp(self) -> List[Dict[int, TensorOrPath]]:
        """
        Compute inverse-Hessian-vector products (IFVP) from gradients and preconditioners.
        Each batch is processed individually and stored separately.

        Returns:
            List of dictionaries mapping batch indices to IFVP tensors or file paths for each layer
        """
        print("Computing inverse-Hessian-vector products (IFVP)...")

        # Load batch information if not already loaded
        if not self.batch_info:
            self._load_batch_info()

        # Initialize result structure
        ifvp: List[Dict[int, TensorOrPath]] = [{} for _ in self.layer_names]

        # Return raw gradients if Hessian type is "none"
        if self.hessian == "none":
            print("Using raw gradients as IFVP since hessian type is 'none'")

            # Simply copy the cached raw gradients structure
            if self.cached_raw_gradients:
                for layer_idx in range(len(self.layer_names)):
                    ifvp[layer_idx] = self.cached_raw_gradients[layer_idx].copy()

            # Store as train_gradients
            self.train_gradients = ifvp
            return ifvp

        # Check if preconditioners are already computed
        if not self.preconditioners:
            print("No preconditioners found. Computing them now...")
            self.compute_preconditioners()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Process each layer
        for layer_idx, _ in tqdm(enumerate(self.layer_names), desc="Processing layers"):
            # Get preconditioner for this layer
            precond_path = self.preconditioners[layer_idx]
            if precond_path is None:
                continue

            # Load preconditioner
            if self.profile and self.profiling_stats:
                start_io = time.time()

            if self.offload == "disk":
                precond = self.io_manager.load_tensor(precond_path)
            else:
                precond = precond_path
                if self.offload == "cpu":
                    precond = precond.to(self.device)

            if self.profile and self.profiling_stats:
                self.profiling_stats.disk_io += time.time() - start_io

            # Process all batches for this layer across all workers
            for worker_id in sorted(self.batch_info.keys()):
                batch_range = self.batch_info[worker_id]['batch_range']
                start_batch, end_batch = batch_range

                # Queue batches to prefetch if using disk IO
                if self.offload == "disk" and isinstance(self.io_manager, DiskIOManager):
                    prefetch_paths = []
                    for batch_idx in range(start_batch, end_batch):
                        grad_path = self._get_file_path(
                            'gradients',
                            layer_idx,
                            batch_idx=batch_idx,
                            worker_id=worker_id
                        )
                        prefetch_paths.append(grad_path)

                    self.io_manager.queue_prefetch(prefetch_paths)

                # Process each batch for this worker
                for batch_idx in range(start_batch, end_batch):
                    # Skip if this batch is already processed
                    if batch_idx in ifvp[layer_idx]:
                        continue

                    # Get gradient path for this batch
                    grad_key = None

                    if self.offload == "disk":
                        grad_key = self._get_file_path(
                            'gradients',
                            layer_idx,
                            batch_idx=batch_idx,
                            worker_id=worker_id
                        )

                        # Skip if file doesn't exist
                        if not os.path.exists(grad_key):
                            continue
                    elif batch_idx in self.cached_raw_gradients[layer_idx]:
                        grad_key = batch_idx
                    else:
                        continue

                    # Load gradient
                    if self.profile and self.profiling_stats:
                        start_io = time.time()

                    if self.offload == "disk":
                        batch_grads = self.io_manager.load_tensor(grad_key)
                    else:
                        batch_grads = self.cached_raw_gradients[layer_idx][grad_key]
                        if self.offload == "cpu":
                            batch_grads = batch_grads.to(self.device)

                    if self.profile and self.profiling_stats:
                        self.profiling_stats.disk_io += time.time() - start_io

                    # Compute IFVP for this batch
                    # Process in smaller chunks if needed to avoid OOM
                    batch_size = min(1024, batch_grads.shape[0])
                    result_tensor = torch.zeros((batch_grads.shape[0], precond.shape[0]),
                                    dtype=precond.dtype)

                    for i in range(0, batch_grads.shape[0], batch_size):
                        end_idx = min(i + batch_size, batch_grads.shape[0])
                        grads_chunk = batch_grads[i:end_idx]

                        # Compute IFVP for this chunk
                        result_chunk = torch.matmul(precond, grads_chunk.t()).t()

                        # Store result
                        result_tensor[i:end_idx] = result_chunk.cpu() if self.cpu_offload else result_chunk

                        # Clean up
                        del grads_chunk, result_chunk
                        torch.cuda.empty_cache()

                    # Save result based on offload strategy
                    if self.offload == "disk":
                        file_path = self._get_file_path(
                            'ifvp',
                            layer_idx,
                            batch_idx=batch_idx,
                            worker_id=worker_id
                        )

                        if self.profile and self.profiling_stats:
                            start_io = time.time()

                        self.io_manager.save_tensor(result_tensor, file_path)

                        if self.profile and self.profiling_stats:
                            self.profiling_stats.disk_io += time.time() - start_io

                        ifvp[layer_idx][batch_idx] = file_path
                    else:
                        # Store in memory
                        ifvp[layer_idx][batch_idx] = result_tensor.cpu() if self.cpu_offload else result_tensor

                    # Clean up
                    del batch_grads, result_tensor
                    torch.cuda.empty_cache()

            # Clean up preconditioner for this layer
            del precond
            torch.cuda.empty_cache()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

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
        Works with batched training gradients.

        Args:
            test_dataloader: DataLoader for test data
            train_dataloader: Optional DataLoader for training data if not cached
            use_cached_ifvp: Whether to use cached IFVP (True) or recompute from cached gradients (False)

        Returns:
            Tensor of influence scores (and profiling stats if profile=True)
        """
        # Load batch information if not already loaded
        if not self.batch_info:
            self._load_batch_info()

        # Validate input
        if train_dataloader is None and self.full_train_dataloader is None and not self.batch_info:
            raise ValueError("No training data provided or cached.")

        # Get IFVP or compute if needed
        if not self.train_gradients or not use_cached_ifvp:
            # Check for cached raw gradients
            if not self.cached_raw_gradients and train_dataloader is not None:
                print("No cached gradients found. Caching from provided dataloader...")
                self.cache_gradients(train_dataloader)

            if not self.preconditioners and self.hessian != "none":
                print("Computing preconditioners...")
                self.compute_preconditioners()

            print("Computing IFVP...")
            self.compute_ifvp()

        # Ensure IFVP is loaded
        if not self.train_gradients:
            raise ValueError("Failed to load or compute IFVP. Check previous steps.")

        # Set up the projectors if not already done
        if self.projectors is None:
            self._setup_projectors(test_dataloader)

        # Compute test gradients using common function with full range
        per_layer_test_gradients, _ = self._compute_gradients(
            test_dataloader,
            is_test=True,
            batch_range=(0, len(test_dataloader))
        )

        torch.cuda.empty_cache()

        # Calculate total training samples
        num_train = 0
        for worker_id in self.batch_info:
            num_train += self.batch_info[worker_id]['total_samples']

        # Initialize influence scores in memory
        num_test = len(test_dataloader.dataset)
        IF_score = torch.zeros(num_train, num_test, device="cpu")

        # Get batch indices for mapping
        test_batch_indices = []
        current_index = 0
        for batch in test_dataloader:
            if isinstance(batch, dict):
                batch_size = next(iter(batch.values())).shape[0]
            else:
                batch_size = batch[0].shape[0]

            col_st = current_index
            col_ed = col_st + batch_size
            test_batch_indices.append((col_st, col_ed))
            current_index += batch_size

        # Remove hooks after collecting test gradients
        if self.hook_manager:
            self.hook_manager.remove_hooks()
            self.hook_manager = None

        # Get mapping of batch_idx to row indices in the final influence matrix
        batch_row_indices = {}
        row_offset = 0
        for worker_id in sorted(self.batch_info.keys()):
            batch_range = self.batch_info[worker_id]['batch_range']
            start_batch, end_batch = batch_range

            # Map each batch index to its row range
            batch_sizes = self.batch_info[worker_id]['sample_counts']
            current_row = row_offset

            for batch_idx, batch_size in enumerate(batch_sizes, start=start_batch):
                batch_row_indices[batch_idx] = (current_row, current_row + batch_size)
                current_row += batch_size

            row_offset = current_row

        # Process each layer
        for layer_idx, layer_name in tqdm(enumerate(self.layer_names), desc="Processing layers"):
            # Skip if no test gradients for this layer
            if not per_layer_test_gradients[layer_idx]:
                continue

            # Get all batch indices for this layer's IFVP
            batch_indices = sorted(self.train_gradients[layer_idx].keys())

            # Prefetch IFVP files if using disk offload
            if self.offload == "disk" and isinstance(self.io_manager, DiskIOManager):
                prefetch_paths = [self.train_gradients[layer_idx][batch_idx]
                                 for batch_idx in batch_indices
                                 if isinstance(self.train_gradients[layer_idx][batch_idx], str)]
                self.io_manager.queue_prefetch(prefetch_paths)

            # Process each training batch
            for batch_idx in tqdm(batch_indices, desc=f"Layer {layer_idx} batches"):
                # Skip if this batch doesn't have row indices
                if batch_idx not in batch_row_indices:
                    continue

                # Get row indices for this batch
                row_st, row_ed = batch_row_indices[batch_idx]

                # Get IFVP for this batch
                ifvp_path = self.train_gradients[layer_idx][batch_idx]

                if self.profile and self.profiling_stats:
                    start_io = time.time()

                if self.offload == "disk":
                    train_grads = self.io_manager.load_tensor(ifvp_path)
                else:
                    train_grads = ifvp_path
                    if self.offload == "cpu":
                        train_grads = train_grads.to(self.device)

                if self.profile and self.profiling_stats:
                    self.profiling_stats.disk_io += time.time() - start_io

                # Process each test batch
                for test_batch_idx in range(len(per_layer_test_gradients[layer_idx])):
                    # Get test gradient
                    if self.offload == "disk":
                        test_path = per_layer_test_gradients[layer_idx][test_batch_idx]
                        if isinstance(test_path, str):
                            test_grad = self.io_manager.load_tensor(test_path)
                        else:
                            test_grad = test_path
                    else:
                        test_grad = per_layer_test_gradients[layer_idx][test_batch_idx]
                        if self.offload == "cpu":
                            test_grad = test_grad.to(self.device)

                    # Get column indices for this test batch
                    col_st, col_ed = test_batch_indices[test_batch_idx]

                    # Compute influence in smaller chunks if needed
                    train_batch_size = min(1024, train_grads.shape[0])
                    for i in range(0, train_grads.shape[0], train_batch_size):
                        end_idx = min(i + train_batch_size, train_grads.shape[0])
                        train_chunk = train_grads[i:end_idx].to(self.device)

                        result = torch.matmul(train_chunk, test_grad.t())

                        # Map local chunk indices to global row indices
                        global_start = row_st + i
                        global_end = row_st + (end_idx - i)
                        IF_score[global_start:global_end, col_st:col_ed] += result.cpu()

                        del train_chunk, result
                        torch.cuda.empty_cache()

                    # Clean up test gradient
                    del test_grad
                    torch.cuda.empty_cache()

                # Clean up train gradients for this batch
                del train_grads
                torch.cuda.empty_cache()

        # Return result
        if self.profile and self.profiling_stats:
            return (IF_score, self.profiling_stats)
        else:
            return IF_score