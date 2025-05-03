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

        # Potentially data subset processing for cache_gradients()
        self.batch_info = {}

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
            data_type: DataTypeOptions,
            layer_idx: int,
            is_temp: bool = False,
            batch_idx: Optional[int] = None,  # Keep for individual batch files
            is_test: bool = False,
            worker_id: Optional[int] = None    # Use for combined files from a worker
        ) -> str:
        """
        Generate a standardized file path for a specific data type and layer.

        Args:
            data_type: Type of data ('gradients', 'preconditioners', or 'ifvp')
            layer_idx: Index of the layer
            is_temp: Whether to use the temporary directory (default: False)
            batch_idx: Optional batch index for individual batch files
            is_test: Whether this is for test data (default: False)
            worker_id: Optional identifier for the worker/job processing a batch group

        Returns:
            Full path to the file
        """
        # Generate the filename based on data type and context
        if batch_idx is not None:
            # Individual batch-specific filename
            prefix = "test_" if is_test else ""
            filename = f"{prefix}layer_{layer_idx}_batch_{batch_idx}.pt"

            # Add worker_id if provided (for individual batch files within a worker)
            if worker_id is not None:
                base, ext = os.path.splitext(filename)
                filename = f"{base}_worker_{worker_id}{ext}"
        else:
            # Map data types to filename patterns for combined files
            type_config = {
                'gradients': f"layer_{layer_idx}_grads.pt",
                'preconditioners': (f"layer_{layer_idx}_precond.pt"
                                if self.hessian == "raw"
                                else f"layer_{layer_idx}_hessian.pt"),
                'ifvp': f"layer_{layer_idx}_ifvp.pt"
            }

            if data_type not in type_config:
                raise ValueError(f"Unknown data type: {data_type}")

            # Add worker_id if provided (for combined files from a worker)
            filename = type_config[data_type]
            if worker_id is not None:
                base, ext = os.path.splitext(filename)
                filename = f"{base}_worker_{worker_id}{ext}"

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

    def _save_tensor(
        self, tensor: torch.Tensor,
        data_type: str,
        layer_idx: int,
        is_temp: bool = True,  # Default to temp for better performance
        save_async: bool = True,  # Default to async for better performance
        batch_idx: Optional[int] = None,
        is_test: bool = False,
        worker_id: Optional[int] = None
    ) -> str:
        """
        Save a tensor to disk with standardized path handling.
        Always saves to temp directory first for better performance.

        Args:
            tensor: The tensor to save
            data_type: Type of data ('gradients', 'preconditioners', or 'ifvp')
            layer_idx: Index of the layer
            is_temp: Whether to save to temp directory (default: True)
            save_async: Whether to save asynchronously using worker threads
            batch_idx: Optional batch index for batch-specific files
            is_test: Whether this is for test data (default: False)
            worker_id: Optional identifier for the batch group when using parallel processing

        Returns:
            Path to the saved file
        """
        if self.profile and self.profiling_stats:
            start_time = time.time()

        # Always get the temp path first for initial fast saving
        temp_path = self._get_file_path(
            data_type, layer_idx, is_temp=True,
            batch_idx=batch_idx, is_test=is_test,
            worker_id=worker_id
        )

        # Save the tensor to temp dir first (for speed)
        if save_async and self.offload == "disk":
            # Queue the save operation for async processing
            self.disk_queue.put((torch.save, (tensor, temp_path), {}))
        else:
            # Save directly
            torch.save(tensor, temp_path)

        # If not using temp and cache_dir is available, schedule async move to permanent location
        if not is_temp and self.cache_dir is not None:
            perm_path = self._get_file_path(
                data_type, layer_idx, is_temp=False,
                batch_idx=batch_idx, is_test=is_test,
                worker_id=worker_id
            )

            # Track that we need to move this file in finalize_cache
            if not hasattr(self, '_pending_finalizations'):
                self._pending_finalizations = []

            self._pending_finalizations.append((temp_path, perm_path, data_type, layer_idx))

        if self.profile and self.profiling_stats:
            self.profiling_stats.disk_io += time.time() - start_time

        # Always return the path where the tensor was saved
        return temp_path if is_temp or self.cache_dir is None else perm_path

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

    def _finalize_cache(
        self,
        data_types: Optional[List[str]] = None,
        worker_id: Optional[int] = None,
        remove_temp: bool = True
    ) -> None:
        """
        Unified method to ensure data is moved from temporary to permanent cache.
        Handles both batch-specific files (when worker_id is provided) and
        global files (when worker_id is None).

        Args:
            data_types: List of data types to finalize (default: all types)
            worker_id: Optional ID of the worker/batch to finalize. If None, finalizes global files
            remove_temp: Whether to remove temporary files after copying to permanent storage
        """
        if self.offload != "disk" or self.cache_dir is None:
            return

        # Default to all data types
        if data_types is None:
            data_types = ['gradients', 'preconditioners', 'ifvp']

        # Message to indicate what we're finalizing
        if worker_id is not None:
            print(f"Finalizing cached data for worker {worker_id}: {data_types}")
        else:
            print(f"Finalizing global cached data: {data_types}")

        # Create required directories
        for data_type in data_types:
            if data_type == 'gradients':
                os.makedirs(os.path.join(self.cache_dir, "grad"), exist_ok=True)
            elif data_type == 'preconditioners':
                subdir = "precond" if self.hessian == "raw" else "hessian"
                os.makedirs(os.path.join(self.cache_dir, subdir), exist_ok=True)
            elif data_type == 'ifvp':
                os.makedirs(os.path.join(self.cache_dir, "ifvp"), exist_ok=True)

        # Handle any pending finalizations from _save_tensor
        # (typically global operations, not batch-specific)
        if hasattr(self, '_pending_finalizations'):
            for temp_path, perm_path, data_type, layer_idx in self._pending_finalizations:
                if data_type in data_types:
                    # If the file still exists in temp and we're finalizing this type
                    if os.path.exists(temp_path):
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(perm_path), exist_ok=True)

                        # Load tensor from temp
                        tensor = self._load_tensor(temp_path)

                        # Save to permanent location
                        torch.save(tensor, perm_path)

                        # Clean up temp file if requested
                        if remove_temp and os.path.exists(temp_path):
                            os.remove(temp_path)

                        # Update references
                        if data_type == 'gradients' and self.cached_raw_gradients:
                            for batch_dict in self.cached_raw_gradients:
                                for wid, path in batch_dict.items():
                                    if path == temp_path:
                                        batch_dict[wid] = perm_path
                        elif data_type == 'preconditioners' and self.preconditioners:
                            for i, path in enumerate(self.preconditioners):
                                if path == temp_path:
                                    self.preconditioners[i] = perm_path
                        elif data_type == 'ifvp' and self.train_gradients:
                            for batch_dict in self.train_gradients:
                                for wid, path in batch_dict.items():
                                    if path == temp_path:
                                        batch_dict[wid] = perm_path

                        # Clean up
                        del tensor

            # Clear the list after processing
            self._pending_finalizations = []

        # Process each layer and its data
        for layer_idx in range(len(self.layer_names)):
            for data_type in data_types:
                # Handle batch-specific files if worker_id is provided
                if worker_id is not None:
                    # First check combined file for this worker_id
                    temp_path = self._get_file_path(
                        data_type, layer_idx, is_temp=True, worker_id=worker_id
                    )

                    if os.path.exists(temp_path):
                        # Get permanent path
                        perm_path = self._get_file_path(
                            data_type, layer_idx, is_temp=False, worker_id=worker_id
                        )

                        # Skip if already exists in permanent location
                        if not os.path.exists(perm_path):
                            # Load tensor from temp
                            tensor = self._load_tensor(temp_path)

                            # Ensure directory exists
                            os.makedirs(os.path.dirname(perm_path), exist_ok=True)

                            # Save to permanent location
                            torch.save(tensor, perm_path)

                            # Update references
                            if data_type == 'gradients' and self.cached_raw_gradients:
                                if layer_idx < len(self.cached_raw_gradients) and worker_id in self.cached_raw_gradients[layer_idx]:
                                    self.cached_raw_gradients[layer_idx][worker_id] = perm_path
                            elif data_type == 'ifvp' and self.train_gradients:
                                if layer_idx < len(self.train_gradients) and worker_id in self.train_gradients[layer_idx]:
                                    self.train_gradients[layer_idx][worker_id] = perm_path

                            # Clean up
                            del tensor

                        # Remove temp file if requested
                        if remove_temp and os.path.exists(temp_path):
                            os.remove(temp_path)

                    # Also handle individual batch files if we have batch info
                    if hasattr(self, 'batch_info') and worker_id in self.batch_info:
                        batch_range = self.batch_info[worker_id]['batch_range']
                        start_batch, end_batch = batch_range

                        for batch_idx in range(start_batch, end_batch):
                            batch_temp_path = self._get_file_path(
                                data_type, layer_idx, is_temp=True, batch_idx=batch_idx
                            )

                            if os.path.exists(batch_temp_path):
                                # Get permanent path
                                batch_perm_path = self._get_file_path(
                                    data_type, layer_idx, is_temp=False, batch_idx=batch_idx
                                )

                                # Skip if already exists
                                if not os.path.exists(batch_perm_path):
                                    # Load tensor from temp
                                    batch_tensor = self._load_tensor(batch_temp_path)

                                    # Ensure directory exists
                                    os.makedirs(os.path.dirname(batch_perm_path), exist_ok=True)

                                    # Save to permanent location
                                    torch.save(batch_tensor, batch_perm_path)

                                    # Clean up
                                    del batch_tensor

                                # Remove temp file if requested
                                if remove_temp and os.path.exists(batch_temp_path):
                                    os.remove(batch_temp_path)

                # Handle global (non-batch-specific) files if worker_id is None
                # or when processing preconditioners (which are always global)
                if worker_id is None or data_type == 'preconditioners':
                    # Check for global file for this layer
                    temp_path = self._get_file_path(data_type, layer_idx, is_temp=True)

                    if os.path.exists(temp_path):
                        # Get permanent path
                        perm_path = self._get_file_path(data_type, layer_idx, is_temp=False)

                        # Skip if already exists in permanent location
                        if not os.path.exists(perm_path):
                            # Load tensor from temp
                            tensor = self._load_tensor(temp_path)

                            # Ensure directory exists
                            os.makedirs(os.path.dirname(perm_path), exist_ok=True)

                            # Save to permanent location
                            torch.save(tensor, perm_path)

                            # Update references
                            if data_type == 'preconditioners' and self.preconditioners:
                                for i, path in enumerate(self.preconditioners):
                                    if path == temp_path:
                                        self.preconditioners[i] = perm_path

                            # Clean up
                            del tensor

                        # Remove temp file if requested
                        if remove_temp and os.path.exists(temp_path):
                            os.remove(temp_path)

        # Free up memory
        torch.cuda.empty_cache()


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
                        # Use the actual batch_idx for uniqueness in filenames
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
        save: bool = False,
        batch_range: Optional[Tuple[int, int]] = None,
        worker_id: Optional[int] = None
    ) -> List[Dict[int, TensorOrPath]]:
        """
        Cache raw projected gradients from training data to disk or memory.
        Works with batched processing - if batch_range and worker_id are not provided,
        processes the entire dataset as a single batch.
        Enhanced with better async I/O operations and immediate cache finalization.

        Args:
            train_dataloader: DataLoader for the training data
            save: Whether to save combined gradients to the cache_dir/grad directory
            batch_range: Optional tuple of (start_batch, end_batch) to process only a subset of batches
            worker_id: Optional identifier for the batch group when using parallel processing

        Returns:
            List of dictionaries mapping worker_id to tensor or filename containing gradients for each layer
        """
        # If batch_range and worker_id are not provided, process the entire dataset
        if batch_range is None and worker_id is None:
            batch_range = (0, len(train_dataloader))
            worker_id = 0
            print(f"No batch information provided, processing entire dataset as worker_id={worker_id}")

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

        # Start disk worker threads if using disk offload
        if self.offload == "disk":
            self._start_disk_workers()

        # Compute gradients using common function with batch range
        per_layer_gradients, batch_sample_counts = self._compute_gradients(
            train_dataloader,
            is_test=False,
            batch_range=batch_range
        )

        # Process collected gradients
        gradients: List[Dict[int, Optional[TensorOrPath]]] = [{} for _ in self.layer_names]

        # Combine gradients for each layer
        for layer_idx, name in enumerate(self.layer_names):
            if not per_layer_gradients[layer_idx]:
                continue

            # Process based on offload strategy
            if self.offload == "none":
                # Concatenate all batches for this layer (already on GPU)
                grads = torch.cat(per_layer_gradients[layer_idx], dim=0)
                gradients[layer_idx][worker_id] = grads
            elif self.offload == "cpu":
                # Concatenate all batches (on CPU)
                grads = torch.cat(per_layer_gradients[layer_idx], dim=0)
                gradients[layer_idx][worker_id] = grads.cpu()
            elif self.offload == "disk":
                # Process from disk files (all in temp directory for speed)
                file_list = per_layer_gradients[layer_idx]

                # Process in chunks to avoid OOM
                chunk_size = min(4096, len(file_list))

                # Track whether we've saved the combined gradients
                combined_saved = False

                # Always save to temp dir first for better performance
                # Process and save combined gradient to temp directory
                for chunk_start in range(0, len(file_list), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(file_list))

                    # Load gradients for this chunk from temp directory
                    chunk_grads = []
                    for file_path in file_list[chunk_start:chunk_end]:
                        grad = self._load_tensor(file_path)
                        chunk_grads.append(grad)

                    if chunk_start == 0:
                        if chunk_end == len(file_list):
                            # If this is the only chunk, save it directly to temp dir
                            combined_tensor = torch.cat(chunk_grads, dim=0)
                            file_path = self._save_tensor(
                                combined_tensor,
                                data_type='gradients',
                                layer_idx=layer_idx,
                                is_temp=True,  # Always save to temp first
                                save_async=True,  # Always use async
                                worker_id=worker_id
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
                                is_temp=True,  # Always save to temp first
                                save_async=False,  # Need sync here since we update this file
                                worker_id=worker_id
                            )
                    elif not combined_saved:
                        # Update the placeholder with this chunk
                        temp_path = self._get_file_path(
                            'gradients',
                            layer_idx,
                            is_temp=True,  # Always use temp path
                            worker_id=worker_id
                        )
                        placeholder = self._load_tensor(temp_path)
                        current_chunk = torch.cat(chunk_grads, dim=0)
                        start_idx = sum(batch_sample_counts[:chunk_start])
                        end_idx = start_idx + current_chunk.shape[0]
                        placeholder[start_idx:end_idx] = current_chunk
                        file_path = self._save_tensor(
                            placeholder,
                            data_type='gradients',
                            layer_idx=layer_idx,
                            is_temp=True,  # Always save to temp first
                            save_async=False,  # Need sync here since we'll update this file
                            worker_id=worker_id
                        )

                    # Clean up memory
                    del chunk_grads
                    if 'placeholder' in locals():
                        del placeholder
                    if 'current_chunk' in locals():
                        del current_chunk

                # Get path to the final saved file
                file_path = self._get_file_path('gradients', layer_idx, is_temp=True, worker_id=worker_id)

                # Reference the file path
                gradients[layer_idx][worker_id] = file_path

        print(f"Cached gradients for {len(self.layer_names)} modules")

        # Store the raw gradients using the new structure
        self.cached_raw_gradients = gradients

        # Store batch information for later
        if not hasattr(self, 'batch_info'):
            self.batch_info = {}

        self.batch_info[worker_id] = {
            'batch_range': batch_range,
            'sample_counts': batch_sample_counts,
            'total_samples': sum(batch_sample_counts)
        }

        # Always save batch info to disk when using batching
        if self.offload == "disk" and self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
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

        # Stop disk workers if they were started
        if self.offload == "disk":
            self._stop_disk_workers()

        if save and self.cache_dir is not None and self.offload == "disk":
            self._finalize_cache(['gradients'], worker_id=worker_id)

        return gradients

    def compute_preconditioners(self, damping: Optional[float] = None, save: bool = False) -> List[TensorOrPath]:
        """
        Compute preconditioners (inverse Hessian) from gradients based on the specified Hessian type.
        Accumulates Hessian contributions from all batches to compute a single preconditioner per layer.
        Enhanced with better async I/O operations.

        Args:
            damping: Damping factor for Hessian inverse (uses self.damping if None)
            save: Whether to save preconditioners to the cache_dir/precond directory

        Returns:
            List of preconditioners for each layer (one preconditioner per layer)
        """
        print(f"Computing preconditioners with hessian type: {self.hessian}...")

        # Load batch information
        if not hasattr(self, 'batch_info') or not self.batch_info:
            # No batch info in memory, try to load from disk
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

        # Use instance damping if not provided
        if damping is None:
            damping = self.damping

        # Start disk worker threads if using disk offload
        if self.offload == "disk":
            self._start_disk_workers()

        # Calculate total samples across all batches
        total_samples = sum(info['total_samples'] for worker_id, info in self.batch_info.items())
        print(f"Total samples across all batches: {total_samples}")

        # Calculate Hessian for each layer (one per layer, not per batch)
        preconditioners: List[Optional[TensorOrPath]] = [None] * len(self.layer_names)

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Process each layer
        for layer_idx, layer_name in tqdm(enumerate(self.layer_names), desc="Processing layers", total=len(self.layer_names)):
            # Initialize Hessian accumulator
            hessian_accumulator = None
            sample_count = 0

            # Get all batches for this layer
            batch_ids = sorted(self.batch_info.keys())

            for worker_id in batch_ids:
                # Try to find gradient file for this batch and layer
                batch_grad_path = None

                # First check cache directory
                if self.offload == "disk" and self.cache_dir is not None:
                    grad_path = self._get_file_path('gradients', layer_idx, is_temp=False, worker_id=worker_id)
                    if os.path.exists(grad_path):
                        batch_grad_path = grad_path

                # Then check temp directory
                if batch_grad_path is None and self.offload == "disk":
                    grad_path = self._get_file_path('gradients', layer_idx, is_temp=True, worker_id=worker_id)
                    if os.path.exists(grad_path):
                        batch_grad_path = grad_path

                if batch_grad_path is None:
                    print(f"Warning: No gradient file found for layer {layer_idx}, batch {worker_id}")
                    continue

                # Load gradient for this batch
                batch_grads = self._load_tensor(batch_grad_path)

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
                        # Save to temp dir first (always), with async=True
                        file_path = self._save_tensor(
                            precond.cpu(),
                            data_type='preconditioners',
                            layer_idx=layer_idx,
                            is_temp=True,  # Always save to temp first
                            save_async=True  # Always use async
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
                        # Save to temp dir first (always), with async=True
                        file_path = self._save_tensor(
                            hessian.cpu(),
                            data_type='preconditioners',
                            layer_idx=layer_idx,
                            is_temp=True,  # Always save to temp first
                            save_async=True  # Always use async
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

        # Wait for all async disk operations to complete
        if self.offload == "disk":
            self.disk_queue.join()

        # Store preconditioners
        self.preconditioners = preconditioners

        # Now that all preconditioners are in temp, move to permanent if requested
        if save and self.cache_dir is not None and self.offload == "disk":
            self._finalize_cache(['preconditioners'])

        return preconditioners

    def compute_ifvp(self, save: bool = False) -> List[Dict[int, TensorOrPath]]:
        """
        Compute inverse-Hessian-vector products (IFVP) from gradients and preconditioners.
        Works with batched gradients but a single preconditioner per layer.
        Enhanced with immediate cache finalization to avoid temp storage limits.

        Args:
            save: Whether to save IFVPs to the cache_dir/ifvp directory

        Returns:
            List of dictionaries mapping worker_id to IFVP tensors or file paths for each layer
        """
        print("Computing inverse-Hessian-vector products (IFVP)...")

        # Load batch information - do this once at the start
        if not hasattr(self, 'batch_info') or not self.batch_info:
            # No batch info in memory, try to load from disk
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

        # Initialize result structure
        ifvp = [{} for _ in self.layer_names]

        # Get batch ids
        batch_ids = sorted(self.batch_info.keys())

        # Return raw gradients if Hessian type is "none"
        if self.hessian == "none":
            print("Using raw gradients as IFVP since hessian type is 'none'")

            # Process each batch
            for worker_id in batch_ids:
                for layer_idx in range(len(self.layer_names)):
                    # Try to find gradient file for this batch and layer
                    batch_grad_path = None

                    # First check cache directory
                    if self.offload == "disk" and self.cache_dir is not None:
                        grad_path = self._get_file_path('gradients', layer_idx, is_temp=False, worker_id=worker_id)
                        if os.path.exists(grad_path):
                            batch_grad_path = grad_path

                    # Then check temp directory
                    if batch_grad_path is None and self.offload == "disk":
                        grad_path = self._get_file_path('gradients', layer_idx, is_temp=True, worker_id=worker_id)
                        if os.path.exists(grad_path):
                            batch_grad_path = grad_path

                    if batch_grad_path is not None:
                        # Since hessian is "none", the gradient is the IFVP
                        # Simply store the path as-is
                        ifvp[layer_idx][worker_id] = batch_grad_path

            # Store as train_gradients
            self.train_gradients = ifvp

            # If save requested, finalize the cache
            if save and self.cache_dir is not None and self.offload == "disk":
                for worker_id in batch_ids:
                    self._finalize_batch_cache(worker_id, ['ifvp'])

            return ifvp

        # Start disk worker threads if using disk offload
        if self.offload == "disk":
            self._start_disk_workers()

        # Initialize preconditioners list
        preconditioners = [None] * len(self.layer_names)

        # Check for preconditioners - first in memory, then on disk
        for layer_idx in range(len(self.layer_names)):
            # Check in memory
            if hasattr(self, 'preconditioners') and self.preconditioners:
                if layer_idx < len(self.preconditioners) and self.preconditioners[layer_idx] is not None:
                    preconditioners[layer_idx] = self.preconditioners[layer_idx]
                    continue

            # Check on disk if using disk offload
            if self.offload == "disk":
                # Check cache dir first
                if self.cache_dir is not None:
                    precond_path = self._get_file_path('preconditioners', layer_idx, is_temp=False)
                    if os.path.exists(precond_path):
                        preconditioners[layer_idx] = precond_path
                        continue

                # Check temp dir
                precond_path = self._get_file_path('preconditioners', layer_idx, is_temp=True)
                if os.path.exists(precond_path):
                    preconditioners[layer_idx] = precond_path
                    continue

        # Check if we found all preconditioners
        if not any(precond is not None for precond in preconditioners):
            print("No preconditioners found. Computing them now...")
            # Compute preconditioners - this will accumulate across all batches
            preconditioners = self.compute_preconditioners(save=save)
        else:
            # Store the found preconditioners
            self.preconditioners = preconditioners

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Process each batch separately, but use the same preconditioner for all batches of a layer
        for worker_id in tqdm(batch_ids, desc="Processing batches"):
            # IMPORTANT CHANGE: Create a worker-specific IFVP dictionary
            worker_ifvp = [{} for _ in self.layer_names]

            # Process each layer for this batch
            for layer_idx, layer_name in enumerate(self.layer_names):
                # Get preconditioner for this layer (same for all batches)
                precond = preconditioners[layer_idx]
                if precond is None:
                    continue

                # Try to find gradient for this batch and layer
                batch_grad_path = None

                # First check for gradient
                if self.offload == "disk":
                    # Check cache directory first
                    if self.cache_dir is not None:
                        grad_path = self._get_file_path('gradients', layer_idx, is_temp=False, worker_id=worker_id)
                        if os.path.exists(grad_path):
                            batch_grad_path = grad_path

                    # Then check temp directory
                    if batch_grad_path is None:
                        grad_path = self._get_file_path('gradients', layer_idx, is_temp=True, worker_id=worker_id)
                        if os.path.exists(grad_path):
                            batch_grad_path = grad_path

                if batch_grad_path is None:
                    print(f"Warning: Missing gradient for layer {layer_idx}, batch {worker_id}")
                    continue

                # Load gradient for this batch
                batch_grads = self._load_tensor(batch_grad_path)

                # Load preconditioner
                if isinstance(precond, str):
                    batch_precond = self._load_tensor(precond).to(self.device)
                else:
                    # Use the in-memory tensor
                    batch_precond = precond.to(self.device)

                # Compute IFVP for this batch
                # Process in smaller chunks if needed to avoid OOM
                batch_size = min(1024, batch_grads.shape[0])
                result_tensor = torch.zeros((batch_grads.shape[0], batch_precond.shape[0]),
                                dtype=batch_precond.dtype)

                for i in range(0, batch_grads.shape[0], batch_size):
                    end_idx = min(i + batch_size, batch_grads.shape[0])
                    grads_chunk = batch_grads[i:end_idx].to(self.device)

                    # Compute IFVP for this chunk
                    result_chunk = torch.matmul(batch_precond, grads_chunk.t()).t()

                    # Store result
                    result_tensor[i:end_idx] = result_chunk.cpu()

                    # Clean up
                    del grads_chunk, result_chunk
                    torch.cuda.empty_cache()

                # Save result based on offload strategy
                if self.offload == "disk":
                    # Always save to temp dir first
                    file_path = self._save_tensor(
                        result_tensor,
                        data_type='ifvp',
                        layer_idx=layer_idx,
                        is_temp=True,  # Save to temp first
                        save_async=True,  # Use async for better performance
                        worker_id=worker_id
                    )
                    worker_ifvp[layer_idx][worker_id] = file_path
                else:
                    # Store in memory
                    worker_ifvp[layer_idx][worker_id] = result_tensor.cpu() if self.cpu_offload else result_tensor

                # Clean up
                del batch_grads, batch_precond, result_tensor
                torch.cuda.empty_cache()

            # Update global IFVP results
            for layer_idx in range(len(self.layer_names)):
                ifvp[layer_idx].update(worker_ifvp[layer_idx])

            if save and self.cache_dir is not None and self.offload == "disk":
                self._finalize_cache(['ifvp'], worker_id=worker_id)

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

        # Wait for all async disk operations to complete
        if self.offload == "disk":
            self.disk_queue.join()

        # Store IFVP results
        self.train_gradients = ifvp

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
        Enhanced with better async I/O operations and improved disk cache checking.

        Args:
            test_dataloader: DataLoader for test data
            train_dataloader: Optional DataLoader for training data if not cached
            use_cached_ifvp: Whether to use cached IFVP (True) or recompute from cached gradients (False)

        Returns:
            Tensor of influence scores (and profiling stats if profile=True)
        """
        # Load batch information - do this once at the start
        if not hasattr(self, 'batch_info') or not self.batch_info:
            # No batch info in memory, try to load from disk
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
                    if train_dataloader is None:
                        raise ValueError("No batch information found and no training dataloader provided.")
            elif train_dataloader is None:
                raise ValueError("No batch information found and no training dataloader provided.")

        # Validate input
        if train_dataloader is None and self.full_train_dataloader is None and not self.batch_info:
            raise ValueError("No training data provided or cached.")

        # Start disk worker threads if using disk offload
        if self.offload == "disk":
            self._start_disk_workers()

        # Get cached IFVP or calculate new ones
        ifvp_train = None

        # First check for cached IFVP on disk if using disk offload
        if use_cached_ifvp and self.offload == "disk" and self.cache_dir is not None:
            # Try to find IFVP files on disk
            ifvp_train = [{} for _ in self.layer_names]
            ifvp_files_found = False

            # Get worker IDs from batch info
            worker_ids = sorted(self.batch_info.keys()) if self.batch_info else []

            for layer_idx in range(len(self.layer_names)):
                for worker_id in worker_ids:
                    # Check cache dir first
                    ifvp_path = self._get_file_path('ifvp', layer_idx, is_temp=False, worker_id=worker_id)
                    if os.path.exists(ifvp_path):
                        ifvp_train[layer_idx][worker_id] = ifvp_path
                        ifvp_files_found = True

                    # Also check temp dir if needed
                    elif self.temp_dir:
                        ifvp_path = self._get_file_path('ifvp', layer_idx, is_temp=True, worker_id=worker_id)
                        if os.path.exists(ifvp_path):
                            ifvp_train[layer_idx][worker_id] = ifvp_path
                            ifvp_files_found = True

            if ifvp_files_found:
                print("Found cached IFVP files on disk. Using them for attribution.")
                self.train_gradients = ifvp_train
            else:
                ifvp_train = None

        # If no IFVP files found on disk, check in-memory cache
        if ifvp_train is None and use_cached_ifvp:
            if hasattr(self, 'train_gradients') and self.train_gradients:
                ifvp_train = self.train_gradients
                print("Using in-memory cached IFVP for attribution.")

        # If still no IFVP, try to compute from cached raw gradients and preconditioners
        if ifvp_train is None:
            print("No cached IFVP found. Attempting to compute from cached gradients...")

            # Check for cached preconditioners
            preconditioners_found = False

            if self.offload == "disk" and self.cache_dir is not None:
                # Check for preconditioner files on disk
                preconditioners = [None] * len(self.layer_names)

                for layer_idx in range(len(self.layer_names)):
                    # Check cache dir first
                    precond_path = self._get_file_path('preconditioners', layer_idx, is_temp=False)
                    if os.path.exists(precond_path):
                        preconditioners[layer_idx] = precond_path
                        preconditioners_found = True

                    # Also check temp dir if needed
                    elif hasattr(self, 'temp_dir') and self.temp_dir:
                        precond_path = self._get_file_path('preconditioners', layer_idx, is_temp=True)
                        if os.path.exists(precond_path):
                            preconditioners[layer_idx] = precond_path
                            preconditioners_found = True

                if preconditioners_found:
                    print("Found cached preconditioners on disk.")
                    self.preconditioners = preconditioners
            elif hasattr(self, 'preconditioners') and self.preconditioners:
                preconditioners_found = True
                print("Using in-memory cached preconditioners.")

            # Check for cached raw gradients
            raw_gradients_found = False

            if self.offload == "disk" and self.cache_dir is not None:
                # Try to find gradient files on disk
                if not hasattr(self, 'cached_raw_gradients') or not self.cached_raw_gradients:
                    self.cached_raw_gradients = [{} for _ in self.layer_names]

                # Get worker IDs from batch info
                worker_ids = sorted(self.batch_info.keys()) if self.batch_info else []

                for layer_idx in range(len(self.layer_names)):
                    for worker_id in worker_ids:
                        # Check cache dir first
                        grad_path = self._get_file_path('gradients', layer_idx, is_temp=False, worker_id=worker_id)
                        if os.path.exists(grad_path):
                            self.cached_raw_gradients[layer_idx][worker_id] = grad_path
                            raw_gradients_found = True

                        # Also check temp dir if needed
                        elif hasattr(self, 'temp_dir') and self.temp_dir:
                            grad_path = self._get_file_path('gradients', layer_idx, is_temp=True, worker_id=worker_id)
                            if os.path.exists(grad_path):
                                self.cached_raw_gradients[layer_idx][worker_id] = grad_path
                                raw_gradients_found = True

                if raw_gradients_found:
                    print("Found cached raw gradients on disk.")
            elif hasattr(self, 'cached_raw_gradients') and self.cached_raw_gradients:
                raw_gradients_found = True
                print("Using in-memory cached raw gradients.")

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

        # Compute test gradients using common function with full range
        per_layer_test_gradients, _ = self._compute_gradients(
            test_dataloader,
            is_test=True,
            batch_range=(0, len(test_dataloader))
        )

        # Calculate total training samples
        num_train = 0
        if self.batch_info:
            for worker_id in self.batch_info:
                num_train += self.batch_info[worker_id]['total_samples']
        else:
            # Handle case if batch_info is missing - this is a fallback
            # but should rarely happen with proper cache handling
            num_train = sum(len(batch) for batch in train_dataloader) if train_dataloader else 0
            if num_train == 0:
                raise ValueError("Cannot determine number of training samples")

        # Initialize influence scores in memory
        num_test = len(test_dataloader.sampler)
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

        # Get mapping of worker_id to row indices in the final influence matrix
        batch_row_indices = {}
        row_offset = 0
        for worker_id in sorted(self.batch_info.keys()):
            batch_size = self.batch_info[worker_id]['total_samples']
            batch_row_indices[worker_id] = (row_offset, row_offset + batch_size)
            row_offset += batch_size

        # Process each layer
        for layer_idx, layer_name in tqdm(enumerate(self.layer_names), desc="Processing layers", total=len(self.layer_names)):
            # Skip if no test gradients for this layer
            if not per_layer_test_gradients[layer_idx]:
                continue

            # Skip if no train gradients for this layer
            if not ifvp_train[layer_idx]:
                continue

            # Process each batch separately
            for worker_id in sorted(ifvp_train[layer_idx].keys()):
                # Skip if this batch doesn't have gradient data
                if worker_id not in batch_row_indices:
                    continue

                # Get row indices for this batch
                row_st, row_ed = batch_row_indices[worker_id]

                # Load train gradients for this batch
                train_grads = None
                if self.offload in ["none", "cpu"]:
                    # Direct tensor access
                    if isinstance(ifvp_train[layer_idx][worker_id], torch.Tensor):
                        train_grads = ifvp_train[layer_idx][worker_id]
                        if self.offload == "cpu":
                            train_grads = train_grads.to(self.device)
                elif self.offload == "disk":
                    # Load from disk
                    if isinstance(ifvp_train[layer_idx][worker_id], str):
                        train_grads = self._load_tensor(ifvp_train[layer_idx][worker_id]).to(self.device)

                if train_grads is None:
                    continue

                # Process each test batch
                for test_batch_idx in range(len(per_layer_test_gradients[layer_idx])):
                    # Get test gradient
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

                    # Compute influence for this batch
                    try:
                        result = torch.matmul(train_grads, test_grad.t())
                        # Update influence scores for this batch's rows
                        IF_score[row_st:row_ed, col_st:col_ed] += result.cpu()
                    except Exception as e:
                        print(f"Error computing influence: {e}")
                        # Fall back to smaller chunks
                        train_batch_size = min(1024, train_grads.shape[0])
                        for i in range(0, train_grads.shape[0], train_batch_size):
                            end_idx = min(i + train_batch_size, train_grads.shape[0])
                            train_chunk = train_grads[i:end_idx]

                            result = torch.matmul(train_chunk, test_grad.t())
                            # Map local chunk indices to global row indices
                            global_start = row_st + i
                            global_end = row_st + end_idx
                            IF_score[global_start:global_end, col_st:col_ed] += result.cpu()

                            del train_chunk, result
                            torch.cuda.empty_cache()

                    # Clean up test gradient
                    del test_grad
                    torch.cuda.empty_cache()

                # Clean up train gradients for this batch
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