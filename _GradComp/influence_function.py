from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Literal, List, Optional, Union, Tuple, TypedDict, cast
import os
import time
import threading
import queue
import tempfile
import shutil
from dataclasses import dataclass

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
        self.cache_dir = cache_dir

        self.full_train_dataloader: Optional[DataLoader] = None
        self.hook_manager: Optional[HookManager] = None
        self.train_gradients: Optional[List[TensorOrPath]] = None
        self.projectors: Optional[List[Any]] = None
        self.cached_raw_gradients: Optional[List[TensorOrPath]] = None
        self.preconditioners: Optional[List[TensorOrPath]] = None

        # Initialize disk offload if needed
        if offload == "disk":
            # Create temp directory for intermediate files
            self.temp_dir = tempfile.mkdtemp()
            print(f"Using temporary directory for disk offload: {self.temp_dir}")

            # Create IFVP directory if specified and it doesn't exist
            if self.cache_dir is not None:
                os.makedirs(self.cache_dir, exist_ok=True)
                print(f"Using custom directory for IFVP files: {self.cache_dir}")

            # Add threading support for disk operations
            self.disk_queue: queue.Queue = queue.Queue()
            self.disk_threads: List[threading.Thread] = []
            self.disk_thread_count = 32

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

    def _start_disk_workers(self) -> None:
        """Start worker threads for handling disk operations"""
        def disk_worker() -> None:
            while True:
                job = self.disk_queue.get()
                if job is None:  # Poison pill to end the thread
                    self.disk_queue.task_done()
                    break

                func, args, kwargs = job
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"Error in disk worker: {e}")
                finally:
                    self.disk_queue.task_done()

        # Start worker threads
        for _ in range(self.disk_thread_count):
            thread = threading.Thread(target=disk_worker, daemon=True)
            thread.start()
            self.disk_threads.append(thread)

    def _stop_disk_workers(self) -> None:
        """Stop all disk worker threads"""
        # Send poison pills to all workers
        for _ in range(self.disk_thread_count):
            self.disk_queue.put(None)

        # Wait for all threads to complete
        for thread in self.disk_threads:
            thread.join()

        self.disk_threads = []

    def _get_file_path(
            self,
            data_type: str,
            layer_idx: int,
            is_temp: bool = False,
            batch_idx: Optional[int] = None,
            is_test: bool = False
        ) -> str:
        """
        Generate a standardized file path for a specific data type and layer.

        Args:
            data_type: Type of data ('gradients', 'preconditioners', or 'ifvp')
            layer_idx: Index of the layer
            is_temp: Whether to use the temporary directory (default: False)
            batch_idx: Optional batch index for batch-specific files
            is_test: Whether this is for test data (default: False)

        Returns:
            Full path to the file
        """
        # Generate the filename based on data type and context
        if batch_idx is not None:
            # Batch-specific filename
            prefix = "test_" if is_test else ""
            filename = f"{prefix}layer_{layer_idx}_batch_{batch_idx}.pt"
        else:
            # Map data types to filename patterns
            type_config = {
                'gradients': f"layer_{layer_idx}_grads.pt",
                'preconditioners': (f"layer_{layer_idx}_precond.pt"
                                if self.hessian == "raw"
                                else f"layer_{layer_idx}_hessian.pt"),
                'ifvp': f"layer_{layer_idx}_ifvp.pt"
            }

            if data_type not in type_config:
                raise ValueError(f"Unknown data type: {data_type}")

            filename = type_config[data_type]

        # Determine the base directory and path
        if is_temp or self.cache_dir is None:
            return os.path.join(self.temp_dir, filename)
        else:
            # Map data types to subdirectory names
            subdir_map = {
                'gradients': 'grad',
                'preconditioners': 'precond' if self.hessian == "raw" else 'hessian',
                'ifvp': 'ifvp'
            }

            if data_type not in subdir_map:
                raise ValueError(f"Unknown data type: {data_type}")

            subdir = subdir_map[data_type]

            # Create the cache subdirectory if it doesn't exist
            cache_subdir = os.path.join(self.cache_dir, subdir)
            os.makedirs(cache_subdir, exist_ok=True)

            return os.path.join(cache_subdir, filename)

    def _get_cached_data(self, data_type: str, layer_idx: int) -> Optional[TensorOrPath]:
        """
        Get cached data, looking first in memory, then in cache dir, then in temp dir.

        Args:
            data_type: Type of data to load ('gradients', 'preconditioners', 'ifvp')
            layer_idx: Index of the layer

        Returns:
            The cached data as tensor, path string, or None if not found
        """
        # 1. First check in-memory storage
        in_memory = None
        if data_type == 'gradients' and self.cached_raw_gradients:
            if layer_idx < len(self.cached_raw_gradients):
                in_memory = self.cached_raw_gradients[layer_idx]
        elif data_type == 'preconditioners' and self.preconditioners:
            if layer_idx < len(self.preconditioners):
                in_memory = self.preconditioners[layer_idx]
        elif data_type == 'ifvp' and self.train_gradients:
            if layer_idx < len(self.train_gradients):
                in_memory = self.train_gradients[layer_idx]

        if in_memory is not None:
            return in_memory

        # Not in memory - try to load from disk if using disk offload
        if self.offload != "disk":
            return None

        # 2. Try cache directory first
        if self.cache_dir is not None:
            cache_path = self._get_file_path(data_type, layer_idx, is_temp=False)
            if os.path.exists(cache_path):
                return cache_path

        # 3. Try temp directory as fallback
        temp_path = self._get_file_path(data_type, layer_idx, is_temp=True)
        if os.path.exists(temp_path):
            return temp_path

        # Not found anywhere
        return None

    def _save_tensor(
            self, tensor: torch.Tensor,
            data_type: str,
            layer_idx: int,
            is_temp: bool = True,
            save_async: bool = True,
            batch_idx: Optional[int] = None,
            is_test: bool = False
        ) -> str:
        """
        Save a tensor to disk with standardized path handling.

        Args:
            tensor: The tensor to save
            data_type: Type of data ('gradients', 'preconditioners', or 'ifvp')
            layer_idx: Index of the layer
            is_temp: Whether to save to temp directory (default: True)
            save_async: Whether to save asynchronously using worker threads
            batch_idx: Optional batch index for batch-specific files
            is_test: Whether this is for test data (default: False)

        Returns:
            Path to the saved file
        """
        if self.profile and self.profiling_stats:
            start_time = time.time()

        # Get the standardized path
        file_path = self._get_file_path(
            data_type, layer_idx, is_temp=is_temp,
            batch_idx=batch_idx, is_test=is_test
        )

        # Save the tensor
        if save_async and self.offload == "disk":
            # Queue the save operation for async processing
            self.disk_queue.put((torch.save, (tensor, file_path), {}))
        else:
            # Save directly
            torch.save(tensor, file_path)

        if self.profile and self.profiling_stats:
            self.profiling_stats.disk_io += time.time() - start_time

        return file_path

    def _load_tensor(self, file_path: str) -> torch.Tensor:
        """
        Load a tensor from disk.

        Args:
            file_path: Path to the tensor file

        Returns:
            The loaded tensor
        """
        if self.profile and self.profiling_stats:
            start_time = time.time()

        # Ensure the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find tensor file: {file_path}")

        # Load the tensor
        tensor = torch.load(file_path)

        if self.profile and self.profiling_stats:
            self.profiling_stats.disk_io += time.time() - start_time

        return tensor

    def _finalize_cache(self, data_types: Optional[List[str]] = None) -> None:
        """
        Ensure all specified data is moved from temporary to permanent cache.

        Args:
            data_types: List of data types to finalize (default: all types)
        """
        if self.offload != "disk" or self.cache_dir is None:
            return

        # Default to all data types
        if data_types is None:
            data_types = ['gradients', 'preconditioners', 'ifvp']

        print(f"Finalizing cached data: {data_types}")

        # Create required directories
        for data_type in data_types:
            if data_type == 'gradients':
                os.makedirs(os.path.join(self.cache_dir, "grad"), exist_ok=True)
            elif data_type == 'preconditioners':
                subdir = "precond" if self.hessian == "raw" else "hessian"
                os.makedirs(os.path.join(self.cache_dir, subdir), exist_ok=True)
            elif data_type == 'ifvp':
                os.makedirs(os.path.join(self.cache_dir, "ifvp"), exist_ok=True)

        # Process each layer and data type
        for layer_idx in range(len(self.layer_names)):
            for data_type in data_types:
                # Get current data reference
                data_ref = self._get_cached_data(data_type, layer_idx)

                if data_ref is None:
                    continue

                # If it's a string path and it's in the temp directory, move to permanent cache
                if isinstance(data_ref, str) and self.temp_dir in data_ref:
                    # Load tensor from temp
                    tensor = self._load_tensor(data_ref)

                    # Save to permanent cache
                    perm_path = self._get_file_path(data_type, layer_idx, is_temp=False)
                    torch.save(tensor, perm_path)

                    # Update the reference in memory
                    if data_type == 'gradients' and self.cached_raw_gradients:
                        self.cached_raw_gradients[layer_idx] = perm_path
                    elif data_type == 'preconditioners' and self.preconditioners:
                        self.preconditioners[layer_idx] = perm_path
                    elif data_type == 'ifvp' and self.train_gradients:
                        self.train_gradients[layer_idx] = perm_path

                    # Clean up
                    del tensor

    def _cleanup(self) -> None:
        """Clean up temporary files and resources"""
        if hasattr(self, 'temp_dir') and self.offload == "disk":
            if os.path.exists(self.temp_dir):
                print(f"Cleaning up temporary files in {self.temp_dir}")
                shutil.rmtree(self.temp_dir)

    def __del__(self) -> None:
        """
        Clean up resources when the object is garbage collected
        """
        self._cleanup()

    def _estimate_memory_requirements(self, dataloader: DataLoader, sample_batch=None) -> Dict[str, int]:
        """
        Estimate memory requirements for gradient storage based on model and data

        Args:
            dataloader: The dataloader to estimate memory for
            sample_batch: Optional sample batch to use for estimation

        Returns:
            Dict of estimated memory requirements per layer in bytes
        """
        # Use a sample batch if not provided
        if sample_batch is None:
            for batch in dataloader:
                sample_batch = batch
                break

        # Create a hook manager
        hook_manager = HookManager(
            self.model,
            self.layer_names,
        )

        # Set projectors in the hook manager
        if self.projectors:
            hook_manager.set_projectors(self.projectors)

        # Zero gradients
        self.model.zero_grad()

        # Prepare inputs
        if isinstance(sample_batch, dict):
            inputs = {k: v.to(self.device) for k, v in sample_batch.items()}
        else:
            inputs = sample_batch[0].to(self.device)

        # Forward pass
        outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

        # Compute loss
        logp = -outputs.loss
        loss = logp - torch.log(1 - torch.exp(logp))

        # Backward pass
        loss.backward()

        # Get projected gradients from hook manager
        with torch.no_grad():
            projected_grads = hook_manager.get_projected_grads()

        # Estimate memory requirements
        memory_estimates = {}
        for idx, name in enumerate(self.layer_names):
            if projected_grads[idx] is not None:
                # Calculate memory required for single gradient
                grad_size = projected_grads[idx].element_size() * projected_grads[idx].nelement()

                # Calculate total memory for all batches
                total_size = grad_size * len(dataloader)
                memory_estimates[name] = total_size
            else:
                memory_estimates[name] = 0

        # Remove hooks
        hook_manager.remove_hooks()

        return memory_estimates

    def _compute_gradients(
        self,
        dataloader: DataLoader,
        is_test: bool = False,
    ) -> List[List[Union[torch.Tensor, str]]]:
        """
        Compute projected gradients for a given dataloader.

        Args:
            dataloader: DataLoader for the data
            is_test_data: Whether this is test data (affects file paths)

        Returns:
            List of lists containing gradients (tensors or file paths) for each layer
        """
        if is_test:
            desc = f"Computing gradients for test data"
        else:
            desc = f"Computing gradients for training data"

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

        # Iterate through the data to compute gradients
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc)):
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
                for idx, grad in enumerate(projected_grads):
                    if grad is None:
                        continue

                    # Detach gradient
                    grad = grad.detach()

                    if self.offload == "none":
                        # Keep on GPU
                        per_layer_gradients[idx].append(grad)
                    elif self.offload == "cpu":
                        # Move to CPU
                        per_layer_gradients[idx].append(grad.cpu())
                        # Free GPU memory
                        del grad
                    elif self.offload == "disk":
                        # Save batch to disk and store the path
                        file_path = self._save_tensor(
                            grad.cpu(),
                            data_type='gradients',
                            layer_idx=idx,
                            is_temp=True,
                            save_async=True,
                            batch_idx=batch_idx,
                            is_test=is_test
                        )
                        per_layer_gradients[idx].append(file_path)
                        # Free GPU memory
                        del grad

            # GPU memory management - ensure we don't run out of memory
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        # Wait for all disk operations to complete
        if self.offload == "disk":
            self.disk_queue.join()

        return per_layer_gradients, batch_sample_counts

    def cache_gradients(
        self,
        train_dataloader: DataLoader,
        save: bool = False
    ) -> List[TensorOrPath]:
        """
        Cache raw projected gradients from training data to disk or memory.

        Args:
            train_dataloader: DataLoader for the training data
            save: Whether to save combined gradients to the cache_dir/grad directory

        Returns:
            List of tensors or filenames containing gradients for each layer
        """
        print(f"Caching gradients from training data with offload strategy: {self.offload}...")
        self.full_train_dataloader = train_dataloader

        # Set up the projectors if not already done
        if self.projectors is None:
            self._setup_projectors(train_dataloader)

        # Estimate memory requirements
        memory_estimates = self._estimate_memory_requirements(train_dataloader)
        total_memory = sum(memory_estimates.values())
        print(f"Estimated memory for gradients: {total_memory / 1e9:.2f} GB")

        # Start disk worker threads if using disk offload
        if self.offload == "disk":
            self._start_disk_workers()

        # Compute gradients using common function
        per_layer_gradients, batch_sample_counts = self._compute_gradients(train_dataloader, is_test=False)

        # Process collected gradients
        gradients: List[Optional[TensorOrPath]] = []

        # Combine gradients for each layer
        for layer_idx, name in enumerate(self.layer_names):
            if not per_layer_gradients[layer_idx]:
                gradients.append(None)
                continue

            # Process based on offload strategy
            if self.offload == "none":
                # Concatenate all batches for this layer (already on GPU)
                grads = torch.cat(per_layer_gradients[layer_idx], dim=0)
                gradients.append(grads)

            elif self.offload == "cpu":
                # Concatenate all batches (on CPU)
                grads = torch.cat(per_layer_gradients[layer_idx], dim=0)
                gradients.append(grads)

            elif self.offload == "disk":
                # Process from disk files (all in temp directory for speed)
                file_list = per_layer_gradients[layer_idx]

                # Process in chunks to avoid OOM
                chunk_size = min(1024, len(file_list))

                # Track whether we've saved the combined gradients
                combined_saved = False

                # Process and save combined gradient to temp directory
                for chunk_start in range(0, len(file_list), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(file_list))

                    # Load gradients for this chunk from temp directory
                    chunk_grads = []
                    for file_path in file_list[chunk_start:chunk_end]:
                        grad = self._load_tensor(file_path)
                        chunk_grads.append(grad)

                    # Save or update combined gradients in temp directory
                    if chunk_start == 0:
                        if chunk_end == len(file_list):
                            # If this is the only chunk, save it directly
                            combined_tensor = torch.cat(chunk_grads, dim=0)
                            file_path = self._save_tensor(
                                combined_tensor,
                                data_type='gradients',
                                layer_idx=layer_idx,
                                is_temp=True,
                                save_async=False
                            )
                            combined_saved = True
                        else:
                            # Otherwise create a placeholder tensor with correct shape
                            sample_shape = chunk_grads[0].shape[1:]
                            total_samples = sum(batch_sample_counts)
                            placeholder = torch.zeros((total_samples, *sample_shape), dtype=chunk_grads[0].dtype)

                            # Update the first part of the placeholder
                            first_chunk = torch.cat(chunk_grads, dim=0)
                            placeholder[:first_chunk.shape[0]] = first_chunk
                            file_path = self._save_tensor(
                                placeholder,
                                data_type='gradients',
                                layer_idx=layer_idx,
                                is_temp=True,
                                save_async=False
                            )
                    elif not combined_saved:
                        # Update the placeholder with this chunk
                        temp_path = self._get_file_path('gradients', layer_idx, is_temp=True)
                        placeholder = self._load_tensor(temp_path)
                        current_chunk = torch.cat(chunk_grads, dim=0)
                        start_idx = sum(batch_sample_counts[:chunk_start])
                        end_idx = start_idx + current_chunk.shape[0]
                        placeholder[start_idx:end_idx] = current_chunk
                        file_path = self._save_tensor(
                            placeholder,
                            data_type='gradients',
                            layer_idx=layer_idx,
                            is_temp=True,
                            save_async=False
                        )

                    # Clean up memory
                    del chunk_grads
                    if 'placeholder' in locals():
                        del placeholder
                    if 'current_chunk' in locals():
                        del current_chunk

                # Reference the file path
                gradients.append(self._get_file_path('gradients', layer_idx, is_temp=True))

        print(f"Cached gradients for {len(self.layer_names)} modules")

        # Store the raw gradients
        self.cached_raw_gradients = cast(List[TensorOrPath], gradients)

        # Remove hooks after collecting all gradients
        if self.hook_manager:
            self.hook_manager.remove_hooks()
            self.hook_manager = None

        # Stop disk workers if they were started
        if self.offload == "disk":
            self._stop_disk_workers()

        # Finalize gradients by moving to custom directory if requested
        if save and self.cache_dir is not None and self.offload == "disk":
            self._finalize_cache(['gradients'])

        return cast(List[TensorOrPath], gradients)

    def compute_preconditioners(self, damping: Optional[float] = None, save: bool = False) -> List[TensorOrPath]:
        """
        Compute preconditioners (inverse Hessian) from gradients based on the specified Hessian type.
        This method centralizes all Hessian type handling in one place.

        Args:
            damping: Damping factor for Hessian inverse (uses self.damping if None)
            save: Whether to save preconditioners to the cache_dir/precond directory

        Returns:
            List of preconditioners for each layer (could be raw Hessian or inverse depending on type)
        """
        if self.hessian == "none":
            print("Skipping preconditioners computation as hessian type is 'none'")
            return self.cached_raw_gradients if self.cached_raw_gradients else []

        print(f"Computing preconditioners with hessian type: {self.hessian}...")

        # Use cached gradients if not provided
        gradients = []
        for layer_idx in range(len(self.layer_names)):
            grad_ref = self._get_cached_data('gradients', layer_idx)
            gradients.append(grad_ref)

        if not any(gradients):
            raise ValueError("No cached gradients available. Call cache_gradients first.")

        # Use instance damping if not provided
        if damping is None:
            damping = self.damping

        # Start disk worker threads if using disk offload
        if self.offload == "disk":
            self._start_disk_workers()

        # Calculate Hessian for each layer
        preconditioners: List[Optional[TensorOrPath]] = []

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        for layer_idx, layer_name in enumerate(self.layer_names):
            grads = gradients[layer_idx]

            if grads is None:
                preconditioners.append(None)
                continue

            # Process based on offload strategy
            if self.offload == "none":
                # Compute Hessian on GPU
                if isinstance(grads, torch.Tensor):
                    hessian = torch.matmul(grads.t(), grads) / len(self.full_train_dataloader.sampler)

                    # Process based on Hessian type
                    if self.hessian == "raw":
                        hessian_inv = stable_inverse(hessian, damping=damping)
                        preconditioners.append(hessian_inv)
                    elif self.hessian in ["kfac", "ekfac"]:
                        preconditioners.append(hessian)
                    else:
                        raise ValueError(f"Unsupported Hessian approximation: {self.hessian}")
                else:
                    preconditioners.append(None)

            elif self.offload == "cpu":
                # Move chunks to GPU, compute partial hessians, then accumulate
                if isinstance(grads, torch.Tensor):
                    hessian_accumulator = None

                    # Process in chunks to avoid OOM
                    chunk_size = min(64, grads.shape[0])
                    for chunk_start in range(0, grads.shape[0], chunk_size):
                        chunk_end = min(chunk_start + chunk_size, grads.shape[0])

                        # Move chunk to GPU
                        chunk_tensor = grads[chunk_start:chunk_end].to(self.device)

                        # Compute partial Hessian
                        chunk_hessian = torch.matmul(chunk_tensor.t(), chunk_tensor)

                        # Accumulate
                        if hessian_accumulator is None:
                            hessian_accumulator = chunk_hessian
                        else:
                            hessian_accumulator += chunk_hessian

                        # Clean up GPU memory
                        del chunk_tensor, chunk_hessian
                        torch.cuda.empty_cache()

                    # Finalize Hessian
                    if hessian_accumulator is not None:
                        hessian = hessian_accumulator / len(self.full_train_dataloader.sampler)

                        # Process based on Hessian type
                        if self.hessian == "raw":
                            hessian_inv = stable_inverse(hessian, damping=damping)
                            preconditioners.append(hessian_inv.cpu() if self.cpu_offload else hessian_inv)
                        elif self.hessian in ["kfac", "ekfac"]:
                            preconditioners.append(hessian.cpu() if self.cpu_offload else hessian)
                        else:
                            raise ValueError(f"Unsupported Hessian approximation: {self.hessian}")

                        del hessian_accumulator
                        torch.cuda.empty_cache()
                    else:
                        preconditioners.append(None)
                else:
                    preconditioners.append(None)

            elif self.offload == "disk":
                # Load from disk if it's a file path
                grads_tensor = None
                if isinstance(grads, str):
                    grads_tensor = self._load_tensor(grads)
                elif isinstance(grads, torch.Tensor):
                    # It's already a tensor
                    grads_tensor = grads

                if grads_tensor is not None:
                    # Accumulate Hessian in chunks
                    hessian_accumulator = None

                    # Process in chunks
                    chunk_size = min(64, grads_tensor.shape[0])
                    for chunk_start in range(0, grads_tensor.shape[0], chunk_size):
                        chunk_end = min(chunk_start + chunk_size, grads_tensor.shape[0])

                        # Move chunk to GPU
                        chunk_tensor = grads_tensor[chunk_start:chunk_end].to(self.device)

                        # Compute partial Hessian
                        chunk_hessian = torch.matmul(chunk_tensor.t(), chunk_tensor)

                        # Accumulate
                        if hessian_accumulator is None:
                            hessian_accumulator = chunk_hessian
                        else:
                            hessian_accumulator += chunk_hessian

                        # Clean up GPU memory
                        del chunk_tensor, chunk_hessian
                        torch.cuda.empty_cache()

                    # Finalize Hessian
                    if hessian_accumulator is not None:
                        hessian = hessian_accumulator / len(self.full_train_dataloader.sampler)

                        # Process based on Hessian type
                        if self.hessian == "raw":
                            hessian_inv = stable_inverse(hessian, damping=damping)

                            # Save preconditioner to disk
                            file_path = self._save_tensor(
                                hessian_inv.cpu(),
                                data_type='preconditioners',
                                layer_idx=layer_idx,
                                is_temp=True,
                                save_async=False
                            )

                            # Reference the file path
                            preconditioners.append(file_path)
                        elif self.hessian in ["kfac", "ekfac"]:
                            # Save Hessian to disk
                            file_path = self._save_tensor(
                                hessian.cpu(),
                                data_type='preconditioners',
                                layer_idx=layer_idx,
                                is_temp=True,
                                save_async=False
                            )

                            # Reference the file path
                            preconditioners.append(file_path)
                        else:
                            raise ValueError(f"Unsupported Hessian approximation: {self.hessian}")

                        # Clean up
                        del hessian_accumulator, hessian
                        if self.hessian == "raw":
                            del hessian_inv
                        torch.cuda.empty_cache()
                    else:
                        preconditioners.append(None)

                    # Clean up
                    del grads_tensor
                else:
                    preconditioners.append(None)

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

        # Stop disk workers if they were started
        if self.offload == "disk":
            self._stop_disk_workers()

        # Store preconditioners
        self.preconditioners = cast(List[TensorOrPath], preconditioners)

        # Finalize preconditioners by moving to custom directory if requested
        if save and self.cache_dir is not None and self.offload == "disk":
            self._finalize_cache(['preconditioners'])

        return cast(List[TensorOrPath], preconditioners)

    def compute_ifvp(self, save: bool = False) -> List[TensorOrPath]:
        """
        Compute inverse-Hessian-vector products (IFVP) from gradients and preconditioners.
        This method relies on compute_preconditioners for all Hessian type handling.

        Args:
            save: Whether to save IFVPs to the cache_dir/ifvp directory

        Returns:
            List of IFVPs for each layer
        """
        # Return raw gradients if Hessian type is "none"
        if self.hessian == "none":
            print("Using raw gradients as IFVP since hessian type is 'none'")
            return self.cached_raw_gradients if self.cached_raw_gradients else []

        print("Computing inverse-Hessian-vector products (IFVP)...")

        # Get gradients
        gradients = []
        for layer_idx in range(len(self.layer_names)):
            grad_ref = self._get_cached_data('gradients', layer_idx)
            gradients.append(grad_ref)

        if not any(gradients):
            raise ValueError("No cached gradients available. Call cache_gradients first.")

        # First check if we have preconditioners cached in memory
        if hasattr(self, 'preconditioners') and self.preconditioners is not None:
            preconditioners = self.preconditioners
        else:
            # Try to load from disk if using disk offload
            if self.offload == "disk":
                preconditioners = []
                for layer_idx in range(len(self.layer_names)):
                    precond_ref = self._get_cached_data('preconditioners', layer_idx)
                    preconditioners.append(precond_ref)

                if not any(preconditioners):
                    # Compute them if not found
                    preconditioners = self.compute_preconditioners()
            else:
                # Compute them if not found
                preconditioners = self.compute_preconditioners()

        # Start disk worker threads if using disk offload
        if self.offload == "disk":
            self._start_disk_workers()

        # Calculate IFVP for each layer
        ifvp: List[Optional[TensorOrPath]] = []

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        for layer_idx, layer_name in enumerate(self.layer_names):
            grads = gradients[layer_idx]
            precond = preconditioners[layer_idx]

            if grads is None or precond is None:
                ifvp.append(None)
                continue

            # Process based on offload strategy
            if self.offload == "none":
                # Calculate IFVP directly on GPU
                if isinstance(grads, torch.Tensor) and isinstance(precond, torch.Tensor):
                    ifvp.append(torch.matmul(precond, grads.t()).t())
                else:
                    ifvp.append(None)

            elif self.offload == "cpu":
                # Process tensors
                if isinstance(grads, torch.Tensor) and isinstance(precond, torch.Tensor):
                    # Move preconditioner to GPU
                    precond_gpu = precond.to(self.device)

                    # Process gradients in batches
                    batch_size = min(1024, grads.shape[0])
                    results = []

                    for i in range(0, grads.shape[0], batch_size):
                        end_idx = min(i + batch_size, grads.shape[0])
                        grads_batch = grads[i:end_idx].to(self.device)

                        # Calculate IFVP for this batch
                        result_batch = torch.matmul(precond_gpu, grads_batch.t()).t()

                        # Move result back to CPU
                        results.append(result_batch.cpu())

                        # Clean up
                        del grads_batch, result_batch
                        torch.cuda.empty_cache()

                    # Combine results
                    ifvp.append(torch.cat(results, dim=0) if results else None)

                    # Clean up
                    del precond_gpu
                    torch.cuda.empty_cache()
                else:
                    ifvp.append(None)

            elif self.offload == "disk":
                # Load data from source files
                grads_tensor = None
                precond_tensor = None

                if isinstance(grads, str):
                    grads_tensor = self._load_tensor(grads)
                elif isinstance(grads, torch.Tensor):
                    grads_tensor = grads

                if isinstance(precond, str):
                    precond_tensor = self._load_tensor(precond).to(self.device)
                elif isinstance(precond, torch.Tensor):
                    precond_tensor = precond.to(self.device)

                if grads_tensor is not None and precond_tensor is not None:
                    # Process gradients in batches
                    batch_size = min(512, grads_tensor.shape[0])

                    # Initialize result tensor
                    result_tensor = torch.zeros((grads_tensor.shape[0], precond_tensor.shape[0]),
                                        dtype=precond_tensor.dtype)

                    # Process in batches
                    for i in range(0, grads_tensor.shape[0], batch_size):
                        end_idx = min(i + batch_size, grads_tensor.shape[0])
                        grads_batch = grads_tensor[i:end_idx].to(self.device)

                        # Calculate IFVP for this batch
                        result_batch = torch.matmul(precond_tensor, grads_batch.t()).t()

                        # Store result
                        result_tensor[i:end_idx] = result_batch.cpu()

                        # Clean up
                        del grads_batch, result_batch
                        torch.cuda.empty_cache()

                    # Save result to disk
                    file_path = self._save_tensor(
                        result_tensor,
                        data_type='ifvp',
                        layer_idx=layer_idx,
                        is_temp=True,
                        save_async=False
                    )

                    # Reference the file path
                    ifvp.append(file_path)

                    # Clean up
                    del precond_tensor, grads_tensor, result_tensor
                    torch.cuda.empty_cache()
                else:
                    ifvp.append(None)

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

        # Stop disk workers if they were started
        if self.offload == "disk":
            self._stop_disk_workers()

        # Store IFVP results
        self.train_gradients = cast(List[TensorOrPath], ifvp)

        # If we have a cache_dir, move files there if requested
        if save and self.cache_dir is not None and self.offload == "disk":
            self._finalize_cache(['ifvp'])

        return cast(List[TensorOrPath], ifvp)

    def cache(self, full_train_dataloader: DataLoader) -> List[TensorOrPath]:
        """
        Cache IFVP for the full training data by leveraging centralized Hessian handling.

        Args:
            full_train_dataloader: DataLoader for the full training data

        Returns:
            List of tensors or filenames containing IFVPs or gradients for each layer
        """
        self.full_train_dataloader = full_train_dataloader
        self.cache_gradients(full_train_dataloader)
        self.compute_ifvp()
        return self.train_gradients

    def attribute(
        self,
        test_dataloader: DataLoader,
        train_dataloader: Optional[DataLoader] = None,
        use_cached_ifvp: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ProfilingStats]]:
        """
        Attribute influence of training examples on test examples.
        Uses centralized Hessian handling through compute_ifvp.

        Args:
            test_dataloader: DataLoader for test data
            train_dataloader: Optional DataLoader for training data if not cached
            use_cached_ifvp: Whether to use cached IFVP (True) or recompute from cached gradients (False)

        Returns:
            Tensor of influence scores (and profiling stats if profile=True)
        """
        # Validate input
        if train_dataloader is None and self.full_train_dataloader is None:
            raise ValueError("No training data provided or cached.")

        # Start disk worker threads if using disk offload
        if self.offload == "disk":
            self._start_disk_workers()

        # Get cached IFVP or calculate new ones
        if train_dataloader is not None and self.full_train_dataloader is None:
            # New train data, cache everything
            num_train = len(train_dataloader.sampler)
            self.full_train_dataloader = train_dataloader

            # Cache gradients first
            self.cache_gradients(train_dataloader)

            # Compute IFVP (handles all Hessian types internally)
            ifvp_train = self.compute_ifvp()
        else:
            # Use cached data
            num_train = len(self.full_train_dataloader.sampler)

            if use_cached_ifvp:
                # Try to use cached IFVP
                if hasattr(self, 'train_gradients') and self.train_gradients is not None:
                    ifvp_train = self.train_gradients
                else:
                    # No cached IFVP in memory, try to find on disk if using disk offload
                    if self.offload == "disk":
                        ifvp_train = []
                        for layer_idx in range(len(self.layer_names)):
                            ifvp_ref = self._get_cached_data('ifvp', layer_idx)
                            ifvp_train.append(ifvp_ref)

                        # If we couldn't find all layers, compute IFVP
                        if not any(ifvp_train):
                            print("No cached IFVP found. Computing from raw gradients...")
                            # Compute IFVP (handles all Hessian types internally)
                            ifvp_train = self.compute_ifvp()
                    else:
                        # No cached IFVP in memory and not using disk offload
                        print("No cached IFVP found. Computing from raw gradients...")
                        # Compute IFVP (handles all Hessian types internally)
                        ifvp_train = self.compute_ifvp()
            else:
                # Force recomputation from raw gradients
                print("Recomputing IFVP from raw gradients as requested...")
                # Compute IFVP (handles all Hessian types internally)
                ifvp_train = self.compute_ifvp()

        # Initialize influence scores in memory
        num_test = len(test_dataloader.sampler)
        IF_score = torch.zeros(num_train, num_test, device="cpu")

        # Compute test gradients using common function
        per_layer_test_gradients, _ = self._compute_gradients(test_dataloader, is_test=True)

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

        for layer_idx in tqdm(range(len(self.layer_names)), desc="Processing layers"):
            # Skip if no test gradients for this layer
            if not per_layer_test_gradients[layer_idx]:
                continue

            # Skip if no train gradients for this layer
            if ifvp_train[layer_idx] is None:
                continue

            # Load train gradients based on offload strategy
            train_grads = None
            if self.offload in ["none", "cpu"]:
                # Load entire train gradients
                if isinstance(ifvp_train[layer_idx], torch.Tensor):
                    train_grads = ifvp_train[layer_idx]
                    if self.offload == "cpu":
                        train_grads = train_grads.to(self.device)
            elif self.offload == "disk":
                # Load entire file from disk
                if isinstance(ifvp_train[layer_idx], str):
                    train_grads = self._load_tensor(ifvp_train[layer_idx]).to(self.device)

            if train_grads is None:
                continue

            # Process test batches
            for test_batch_idx in range(len(per_layer_test_gradients[layer_idx])):
                # Load test gradient
                test_grad = None
                if self.offload in ["none", "cpu"]:
                    test_grad = per_layer_test_gradients[layer_idx][test_batch_idx]
                    if self.offload == "cpu" and isinstance(test_grad, torch.Tensor):
                        test_grad = test_grad.to(self.device)
                elif self.offload == "disk":
                    if isinstance(per_layer_test_gradients[layer_idx][test_batch_idx], str):
                        test_grad = self._load_tensor(per_layer_test_gradients[layer_idx][test_batch_idx]).to(self.device)

                if test_grad is None:
                    continue

                # Get column indices for this test batch
                col_st, col_ed = test_batch_indices[test_batch_idx]

                # Compute influence for the entire train set at once
                try:
                    result = torch.matmul(train_grads, test_grad.t())
                    # Update influence scores
                    IF_score[:, col_st:col_ed] += result.cpu()
                except Exception as e:
                    print(f"Error computing influence: {e}")
                    # If we hit memory issues, fall back to batched approach
                    print("Falling back to batched approach...")
                    train_batch_size = min(4096, train_grads.shape[0])
                    for train_batch_start in range(0, train_grads.shape[0], train_batch_size):
                        train_batch_end = min(train_batch_start + train_batch_size, train_grads.shape[0])
                        train_batch = train_grads[train_batch_start:train_batch_end]

                        result = torch.matmul(train_batch, test_grad.t())
                        IF_score[train_batch_start:train_batch_end, col_st:col_ed] += result.cpu()

                        del train_batch, result
                        torch.cuda.empty_cache()

                # Clean up test gradient
                del test_grad
                torch.cuda.empty_cache()

            # Clean up train gradients
            del train_grads
            torch.cuda.empty_cache()

        # Stop disk workers if they were started
        if self.offload == "disk":
            self._stop_disk_workers()

        # Return result
        if self.profile and self.profiling_stats:
            return (IF_score, self.profiling_stats)
        else:
            return IF_score