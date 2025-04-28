from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Literal
import os
import time
import threading
import queue
import tempfile
import shutil

if TYPE_CHECKING:
    from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from .hook import HookManager
from .utils import stable_inverse
from .projector import setup_model_projectors

class IFAttributor:
    """
    Optimized influence function calculator using hooks for efficient gradient projection.
    Works with standard PyTorch layers with support for disk-based offloading.
    """

    def __init__(
        self,
        setting: str,
        model: nn.Module,
        layer_names: Union[str, List[str]],
        hessian: Literal["none", "raw", "kfac", "ekfac"] = "raw",
        damping: float = None,
        profile: bool = False,
        device: str = 'cpu',
        projector_kwargs: Dict = None,
        offload: Literal["none", "cpu", "disk"] = "none",
        cache_dir: str = None,
    ) -> None:
        """
        Optimized Influence Function Attributor.

        Args:
            setting (str): The setting of the experiment
            model (nn.Module): PyTorch model.
            layer_names (List[str]): Names of layers to attribute.
            hessian (str): Type of Hessian approximation ("none", "raw", "kfac", "ekfac"). Defaults to "raw".
            damping (float): Damping used when calculating the Hessian inverse. Defaults to None.
            profile (bool): Record time used in various parts of the algorithm run. Defaults to False.
            device (str): Device to run the model on. Defaults to 'cpu'.
            offload (str): Memory management strategy ("none", "cpu", "disk"). Defaults to "none".
            projector_kwargs (Dict): Keyword arguments for projector. Defaults to None.
            cache_dir (str): Directory to save final IFVP files. Only used when offload="disk". Defaults to None.
        """
        self.setting = setting
        self.model = model
        self.model.to(device)
        self.model.eval()

        # Ensure layer_names is a list
        if isinstance(layer_names, str):
            self.layer_names = [layer_names]
        else:
            self.layer_names = layer_names

        self.hessian = hessian
        self.damping = damping
        self.profile = profile
        self.device = device

        # Validate and set offload strategy
        valid_offload_strategies = ["none", "cpu", "disk"]
        if offload not in valid_offload_strategies:
            raise ValueError(f"Invalid offload strategy: {offload}. Must be one of {valid_offload_strategies}")
        self.offload = offload

        # Keep cpu_offload for backwards compatibility
        self.cpu_offload = offload in ["cpu", "disk"]

        # Store the IFVP directory
        self.cache_dir = cache_dir

        self.projector_kwargs = projector_kwargs or {}

        self.full_train_dataloader = None
        self.hook_manager = None
        self.train_gradients = None
        self.projectors = None

        # Disk offload paths
        if offload == "disk":
            # Create temp directory for intermediate files
            self.temp_dir = tempfile.mkdtemp()
            print(f"Using temporary directory for disk offload: {self.temp_dir}")

            # Create IFVP directory if specified and it doesn't exist
            if self.cache_dir is not None:
                os.makedirs(self.cache_dir, exist_ok=True)
                print(f"Using custom directory for IFVP files: {self.cache_dir}")

            # Add threading support for disk operations
            self.disk_queue = queue.Queue()
            self.disk_threads = []
            self.disk_thread_count = 4  # Number of threads for parallel disk operations

        # Initialize profiling stats
        if self.profile:
            self.profiling_stats = {
                'projection': 0.0,
                'forward': 0.0,
                'backward': 0.0,
                'precondition': 0.0,
                'disk_io': 0.0,
            }

    def _setup_projectors(self, train_dataloader: torch.utils.data.DataLoader) -> None:
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

    def _start_disk_workers(self):
        """Start worker threads for handling disk operations"""

        def disk_worker():
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

    def _stop_disk_workers(self):
        """Stop all disk worker threads"""
        # Send poison pills to all workers
        for _ in range(self.disk_thread_count):
            self.disk_queue.put(None)

        # Wait for all threads to complete
        for thread in self.disk_threads:
            thread.join()

        self.disk_threads = []

    def _save_tensor_to_disk(self, tensor, filename):
        """Save a tensor to disk"""
        if self.profile:
            start_time = time.time()

        # Determine the directory to use
        # Use custom directory for IFVP files if specified
        if self.cache_dir is not None and filename.startswith("layer_") and filename.endswith("_ifvp.pt"):
            full_path = os.path.join(self.cache_dir, filename)
        else:
            full_path = os.path.join(self.temp_dir, filename)

        torch.save(tensor, full_path)

        if self.profile:
            self.profiling_stats['disk_io'] += time.time() - start_time

    def _load_tensor_from_disk(self, filename):
        """
        Load a tensor from disk.
        """
        if self.profile:
            start_time = time.time()

        # Try loading from IFVP directory first if applicable
        if self.cache_dir is not None and filename.startswith("layer_") and filename.endswith("_ifvp.pt"):
            full_path = os.path.join(self.cache_dir, filename)
            if not os.path.exists(full_path):
                # Fall back to temp directory if file not found
                full_path = os.path.join(self.temp_dir, filename)
        else:
            full_path = os.path.join(self.temp_dir, filename)

        # Load the tensor
        tensor = torch.load(full_path)

        if self.profile:
            self.profiling_stats['disk_io'] += time.time() - start_time

        return tensor

    def _update_tensor_on_disk(self, filename, new_data, slice_idx=None):
        """
        Update part of a tensor stored on disk.

        Args:
            filename: The name of the file to update
            new_data: The new data to insert
            slice_idx: Indices for slicing (tuple or tuple of tuples)
        """
        if self.profile:
            start_time = time.time()

        full_path = os.path.join(self.temp_dir, filename)

        # Load the tensor
        tensor = torch.load(full_path)

        # Apply the update
        if slice_idx is None:
            # Replace the entire tensor
            tensor = new_data
        elif isinstance(slice_idx[0], tuple):
            # Multi-dimensional slice
            slices = tuple(slice(start, end) for start, end in slice_idx)
            tensor[slices] = new_data
        else:
            # Single-dimensional slice
            start, end = slice_idx
            tensor[start:end] = new_data

        # Save the updated tensor
        torch.save(tensor, full_path)

        if self.profile:
            self.profiling_stats['disk_io'] += time.time() - start_time

    def _queue_tensor_save(self, tensor, filename):
        """Queue a tensor to be saved to disk by a worker thread"""
        self.disk_queue.put((self._save_tensor_to_disk, (tensor, filename), {}))

    def _finalize_ifvp_files(self):
        """
        Move IFVP files to the custom directory if specified.
        Called at the end of the cache method.
        """
        if self.offload != "disk" or self.cache_dir is None:
            return

        # Find all IFVP files in the temp directory
        for layer_id in range(len(self.layer_names)):
            ifvp_filename = f"layer_{layer_id}_ifvp.pt"
            temp_path = os.path.join(self.temp_dir, ifvp_filename)

            if os.path.exists(temp_path):
                # Copy to the custom directory
                target_path = os.path.join(self.cache_dir, ifvp_filename)
                print(f"Moving IFVP file to custom directory: {target_path}")
                shutil.copy2(temp_path, target_path)

                # Update reference in train_gradients if it's a filename
                if isinstance(self.train_gradients[layer_id], str):
                    self.train_gradients[layer_id] = target_path

    def _estimate_memory_requirements(self, dataloader, sample_batch=None):
        """
        Estimate memory requirements for gradient storage based on model and data

        Args:
            dataloader: The dataloader to estimate memory for
            sample_batch: Optional sample batch to use for estimation

        Returns:
            dict: Estimated memory requirements per layer in bytes
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

    def _cleanup(self):
        """Clean up temporary files and resources"""
        if hasattr(self, 'temp_dir') and self.offload == "disk":
            if os.path.exists(self.temp_dir):
                print(f"Cleaning up temporary files in {self.temp_dir}")
                shutil.rmtree(self.temp_dir)

    def __del__(self):
        """
        Clean up resources when the object is garbage collected
        """
        self._cleanup()

    def cache(
        self,
        full_train_dataloader: torch.utils.data.DataLoader
    ) -> List[torch.Tensor]:
        """
        Cache IFVP for the full training data.

        Args:
            full_train_dataloader: DataLoader for the full training data
        """
        print(f"Extracting information from training data with offload strategy: {self.offload}...")
        self.full_train_dataloader = full_train_dataloader

        # Set up the projectors
        if self.projectors is None:
            self._setup_projectors(full_train_dataloader)

        # Estimate memory requirements
        memory_estimates = self._estimate_memory_requirements(full_train_dataloader)
        total_memory = sum(memory_estimates.values())
        print(f"Estimated memory for gradients: {total_memory / 1e9:.2f} GB")

        # Start disk worker threads if using disk offload
        if self.offload == "disk":
            self._start_disk_workers()

        # Initialize storage for gradients
        total_batches = len(full_train_dataloader)
        gpu_batch_size = min(32, total_batches)  # Process this many batches before offloading

        # Create hook manager
        self.hook_manager = HookManager(
            self.model,
            self.layer_names,
        )

        # Set projectors in the hook manager
        if self.projectors:
            self.hook_manager.set_projectors(self.projectors)

        # Preallocate gradient storage based on offload strategy
        per_layer_gradients = []
        for layer_idx, name in enumerate(self.layer_names):
            if self.offload == "none":
                # Store all on GPU
                per_layer_gradients.append([])
            elif self.offload in ["cpu", "disk"]:
                # For CPU and disk offload, we'll use a list with either tensor chunks or filenames
                per_layer_gradients.append([])

        # Collect sample sizes for each batch
        batch_sample_counts = []

        # Batch counter for GPU offloading
        current_gpu_batch = 0

        # Iterate through the training data to compute gradients
        for train_batch_idx, train_batch in enumerate(tqdm(full_train_dataloader, desc="Processing training data")):
            # Zero gradients
            self.model.zero_grad()

            # Prepare inputs
            if isinstance(train_batch, dict):
                inputs = {k: v.to(self.device) for k, v in train_batch.items()}
                batch_size = next(iter(train_batch.values())).shape[0]
            else:
                inputs = train_batch[0].to(self.device)
                batch_size = train_batch[0].shape[0]

            batch_sample_counts.append(batch_size)

            # Forward pass
            if self.profile:
                torch.cuda.synchronize(self.device)
                start_time = time.time()

            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['forward'] += time.time() - start_time

            # Compute custom loss
            logp = -outputs.loss
            train_loss = logp - torch.log(1 - torch.exp(logp))

            # Backward pass
            if self.profile:
                torch.cuda.synchronize(self.device)
                start_time = time.time()

            train_loss.backward()

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['backward'] += time.time() - start_time

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
                        # Save to disk using unique filename
                        filename = f"layer_{idx}_batch_{train_batch_idx}.pt"
                        # Add to our tracking list
                        per_layer_gradients[idx].append(filename)
                        # Queue for saving by worker thread
                        self._queue_tensor_save(grad.cpu(), filename)
                        # Free GPU memory
                        del grad

            # GPU memory management - ensure we don't run out of memory
            if train_batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # Update GPU batch counter
            current_gpu_batch += 1

            # Force offload and reset counter if needed
            if current_gpu_batch >= gpu_batch_size and self.offload in ["cpu", "disk"]:
                current_gpu_batch = 0
                # Force offload
                torch.cuda.empty_cache()

        # Wait for all disk operations to complete
        if self.offload == "disk":
            self.disk_queue.join()

        # Remove hooks after collecting all gradients
        self.hook_manager.remove_hooks()

        # Process all collected gradients
        hessians = []
        train_grads = []

        # Time for precondition calculations
        if self.profile:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Calculate Hessian and prepare gradients for each layer
        for layer_idx, name in enumerate(self.layer_names):
            if not per_layer_gradients[layer_idx]:
                train_grads.append(None)
                hessians.append(None)
                continue

            # Process based on offload strategy
            if self.offload == "none":
                # Concatenate all batches for this layer (already on GPU)
                grads = torch.cat(per_layer_gradients[layer_idx], dim=0)

                # Compute Hessian on GPU
                hessian = torch.matmul(grads.t(), grads) / len(full_train_dataloader.sampler)

                # Store for later use
                train_grads.append(grads)
                hessians.append(hessian)

            elif self.offload == "cpu":
                # Move chunks to GPU, compute partial hessians, then accumulate
                hessian_accumulator = None
                grads_list = per_layer_gradients[layer_idx]

                # Process in chunks to avoid OOM
                chunk_size = min(64, len(grads_list))
                for chunk_start in range(0, len(grads_list), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(grads_list))

                    # Move chunk to GPU
                    chunk_grads = [grad.to(self.device) for grad in grads_list[chunk_start:chunk_end]]
                    chunk_tensor = torch.cat(chunk_grads, dim=0)

                    # Compute partial Hessian
                    chunk_hessian = torch.matmul(chunk_tensor.t(), chunk_tensor)

                    # Accumulate
                    if hessian_accumulator is None:
                        hessian_accumulator = chunk_hessian
                    else:
                        hessian_accumulator += chunk_hessian

                    # Clean up GPU memory
                    del chunk_grads, chunk_tensor, chunk_hessian
                    torch.cuda.empty_cache()

                # Finalize Hessian
                if hessian_accumulator is not None:
                    hessian = hessian_accumulator / len(full_train_dataloader.sampler)
                    hessians.append(hessian.cpu() if self.cpu_offload else hessian)
                    del hessian_accumulator
                    torch.cuda.empty_cache()
                else:
                    hessians.append(None)

                # Store gradients
                train_grads.append(torch.cat([grad for grad in grads_list], dim=0))

            elif self.offload == "disk":
                # Process from disk files
                file_list = per_layer_gradients[layer_idx]
                hessian_accumulator = None

                # Process in chunks to avoid OOM
                chunk_size = min(32, len(file_list))

                # Create a combined filename for the full gradients
                combined_grad_filename = f"layer_{layer_idx}_combined_grads.pt"

                # Track whether we've saved the combined gradients
                combined_saved = False

                # Process and accumulate in chunks
                for chunk_start in range(0, len(file_list), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(file_list))

                    # Load gradients for this chunk
                    chunk_grads = []

                    for filename in file_list[chunk_start:chunk_end]:
                        grad = self._load_tensor_from_disk(filename)
                        chunk_grads.append(grad.to(self.device))

                    # Concatenate chunk
                    chunk_tensor = torch.cat(chunk_grads, dim=0)

                    # Compute partial Hessian
                    chunk_hessian = torch.matmul(chunk_tensor.t(), chunk_tensor)

                    # Accumulate
                    if hessian_accumulator is None:
                        hessian_accumulator = chunk_hessian
                    else:
                        hessian_accumulator += chunk_hessian

                    # Save combined gradients
                    if chunk_start == 0:
                        if chunk_end == len(file_list):
                            # If this is the only chunk, save it directly
                            combined_tensor = torch.cat(chunk_grads, dim=0).cpu()
                            self._save_tensor_to_disk(combined_tensor, combined_grad_filename)
                            combined_saved = True
                        else:
                            # Otherwise create a placeholder tensor with correct shape
                            sample_shape = chunk_grads[0].shape[1:]
                            total_samples = sum(batch_sample_counts)
                            placeholder = torch.zeros((total_samples, *sample_shape), dtype=chunk_grads[0].dtype)
                            self._save_tensor_to_disk(placeholder, combined_grad_filename)

                            # Update the first part of the placeholder
                            first_chunk = torch.cat(chunk_grads, dim=0).cpu()
                            placeholder[:first_chunk.shape[0]] = first_chunk
                            self._save_tensor_to_disk(placeholder, combined_grad_filename)
                    elif not combined_saved:
                        # Update the placeholder with this chunk
                        placeholder = self._load_tensor_from_disk(combined_grad_filename)
                        current_chunk = torch.cat(chunk_grads, dim=0).cpu()
                        start_idx = sum(batch_sample_counts[:chunk_start])
                        end_idx = start_idx + current_chunk.shape[0]
                        placeholder[start_idx:end_idx] = current_chunk
                        self._save_tensor_to_disk(placeholder, combined_grad_filename)

                    # Clean up GPU memory
                    del chunk_grads, chunk_tensor, chunk_hessian
                    torch.cuda.empty_cache()

                # Finalize Hessian
                if hessian_accumulator is not None:
                    hessian = hessian_accumulator / len(full_train_dataloader.sampler)

                    # Save Hessian to disk
                    hessian_filename = f"layer_{layer_idx}_hessian.pt"
                    self._save_tensor_to_disk(hessian.cpu(), hessian_filename)

                    # Reference by filename
                    hessians.append(hessian_filename)

                    del hessian_accumulator, hessian
                    torch.cuda.empty_cache()
                else:
                    hessians.append(None)

                # Reference gradients by filename
                train_grads.append(combined_grad_filename)

        print(f"Computed gradient covariance for {len(self.layer_names)} modules")

        # Calculate inverse Hessian vector products
        if self.profile:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        if self.hessian == "none":
            self.train_gradients = train_grads
        elif self.hessian == "raw":
            print("Computing gradient covariance inverse...")

            ifvp_train = []

            # Process each layer
            for layer_id, layer_name in enumerate(self.layer_names):
                if self.offload == "none" or self.offload == "cpu":
                    grads = train_grads[layer_id]
                    hessian = hessians[layer_id]

                    if grads is None or hessian is None:
                        ifvp_train.append(None)
                        continue

                    # Process Hessian inverse and IFVP calculation
                    if self.cpu_offload:
                        # Move Hessian to GPU for inverse calculation
                        hessian_gpu = hessian.to(device=self.device)

                        # Calculate inverse on GPU (more efficient)
                        hessian_inv = stable_inverse(hessian_gpu, damping=self.damping)

                        # Process gradients in batches to avoid memory issues
                        batch_size = min(1024, grads.shape[0])  # Adjust based on available memory
                        results = []

                        for i in range(0, grads.shape[0], batch_size):
                            end_idx = min(i + batch_size, grads.shape[0])
                            grads_batch = grads[i:end_idx].to(device=self.device)

                            # Calculate IFVP for this batch
                            result_batch = torch.matmul(hessian_inv, grads_batch.t()).t()

                            # Move result back to CPU
                            results.append(result_batch.cpu())

                        # Combine results from all batches
                        ifvp_train.append(torch.cat(results, dim=0) if results else None)
                    else:
                        # Calculate IFVP directly on GPU
                        hessian_inv = stable_inverse(hessian, damping=self.damping)
                        ifvp_train.append(torch.matmul(hessian_inv, grads.t()).t())

                elif self.offload == "disk":
                    # File paths to load data
                    grads_filename = train_grads[layer_id]
                    hessian_filename = hessians[layer_id]

                    if grads_filename is None or hessian_filename is None:
                        ifvp_train.append(None)
                        continue

                    # File path to save results
                    ifvp_filename = f"layer_{layer_id}_ifvp.pt"

                    # Load Hessian to GPU and compute inverse
                    hessian = self._load_tensor_from_disk(hessian_filename).to(device=self.device)
                    hessian_inv = stable_inverse(hessian, damping=self.damping)

                    # Process gradients in batches
                    grads = self._load_tensor_from_disk(grads_filename)
                    batch_size = min(512, grads.shape[0])

                    # Initialize placeholder with the correct shape
                    # The shape should be (batch_size, hessian dimension)
                    combined_ifvp = torch.zeros((grads.shape[0], hessian_inv.shape[0]), dtype=hessian_inv.dtype)

                    # Process in batches
                    for i in range(0, grads.shape[0], batch_size):
                        end_idx = min(i + batch_size, grads.shape[0])
                        grads_batch = grads[i:end_idx].to(device=self.device)

                        # Calculate IFVP for this batch
                        result_batch = torch.matmul(hessian_inv, grads_batch.t()).t()

                        # Update combined tensor
                        combined_ifvp[i:end_idx] = result_batch.cpu()

                        # Clean up
                        del grads_batch, result_batch
                        torch.cuda.empty_cache()

                    # Save the combined IFVP tensor
                    self._save_tensor_to_disk(combined_ifvp, ifvp_filename)

                    # Add filename reference
                    ifvp_train.append(ifvp_filename)

                    # Clean up
                    del hessian, hessian_inv
                    torch.cuda.empty_cache()

            print(f"Computed gradient covariance inverse for {len(self.layer_names)} modules")

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['precondition'] += time.time() - start_time

            self.train_gradients = ifvp_train
        else:
            raise ValueError(f"Unsupported Hessian approximation: {self.hessian}")

        # Stop disk workers if they were started
        if self.offload == "disk":
            self._stop_disk_workers()

            self._finalize_ifvp_files()

        # Return gradients (or references to them)
        return self.train_gradients

    def _calculate_ifvp(self, train_dataloader: torch.utils.data.DataLoader) -> List[torch.Tensor]:
        """
        Calculate IFVP for the training data.

        Args:
            train_dataloader: DataLoader for training data

        Returns:
            List of tensors containing IFVP for each layer
        """
        # Check if gradients are already cached
        if self.train_gradients is not None:
            return self.train_gradients

        # Cache the gradients
        return self.cache(train_dataloader)

    def attribute(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Attribute influence of training examples on test examples.

        Args:
            test_dataloader: DataLoader for test data
            train_dataloader: Optional DataLoader for training data if not cached

        Returns:
            Tensor of influence scores (and profiling stats if profile=True)
        """
        if self.full_train_dataloader is not None and train_dataloader is not None:
            raise ValueError(
                "You have cached a training loader by .cache() and you are trying to attribute "
                "a different training loader. If this new training loader is a subset of the cached "
                "training loader, please don't input the training dataloader in .attribute() and "
                "directly use index to select the corresponding scores."
            )

        if train_dataloader is None and self.full_train_dataloader is None:
            raise ValueError(
                "You did not state a training loader in .attribute() and you did not cache a "
                "training loader by .cache(). Please provide a training loader or cache a "
                "training loader."
            )

        # Start disk worker threads if using disk offload
        if self.offload == "disk":
            self._start_disk_workers()

        # Use cached IFVP or calculate new ones
        if train_dataloader is not None and self.full_train_dataloader is None:
            num_train = len(train_dataloader.sampler)
            ifvp_train = self._calculate_ifvp(train_dataloader)
        else:
            num_train = len(self.full_train_dataloader.sampler)
            ifvp_train = self.train_gradients

        # Initialize influence scores in memory
        num_test = len(test_dataloader.sampler)
        IF_score = torch.zeros(num_train, num_test, device="cpu")

        # Create hook manager for test examples
        self.hook_manager = HookManager(
            self.model,
            self.layer_names,
        )

        # Set projectors in the hook manager if available
        if self.projectors:
            self.hook_manager.set_projectors(self.projectors)

        # Collect test gradients first
        per_layer_test_gradients = [[] for _ in self.layer_names]
        test_batch_indices = []

        # Process each test batch
        print("Collecting projected gradients from test data...")
        for test_batch_idx, test_batch in enumerate(tqdm(test_dataloader, desc="Collecting test gradients")):
            # Zero gradients
            self.model.zero_grad()

            # Prepare inputs
            if isinstance(test_batch, dict):
                inputs = {k: v.to(self.device) for k, v in test_batch.items()}
                batch_size = next(iter(test_batch.values())).shape[0]
            else:
                inputs = test_batch[0].to(self.device)
                batch_size = test_batch[0].shape[0]

            # Forward pass
            if self.profile:
                torch.cuda.synchronize(self.device)
                start_time = time.time()

            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['forward'] += time.time() - start_time

            # Compute loss
            logp = -outputs.loss
            test_loss = logp - torch.log(1 - torch.exp(logp))

            # Backward pass
            if self.profile:
                torch.cuda.synchronize(self.device)
                start_time = time.time()

            test_loss.backward()

            if self.profile:
                torch.cuda.synchronize(self.device)
                self.profiling_stats['backward'] += time.time() - start_time

            # Get projected gradients from hook manager
            with torch.no_grad():
                projected_grads = self.hook_manager.get_projected_grads()

                # Store test batch indices for later mapping
                col_st = test_batch_idx * batch_size
                col_ed = min(col_st + batch_size, num_test)
                test_batch_indices.append((col_st, col_ed))

                # Collect test gradients
                for idx, grad in enumerate(projected_grads):
                    if grad is not None:
                        # Detach gradient
                        grad = grad.detach()

                        if self.offload == "none":
                            per_layer_test_gradients[idx].append(grad)
                        elif self.offload == "cpu":
                            per_layer_test_gradients[idx].append(grad.cpu())
                            del grad
                        elif self.offload == "disk":
                            # Save to disk with unique name
                            filename = f"test_layer_{idx}_batch_{test_batch_idx}.pt"
                            self._save_tensor_to_disk(grad.cpu(), filename)
                            per_layer_test_gradients[idx].append(filename)
                            del grad

            # GPU memory management
            if test_batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        # Wait for disk operations to complete if using disk offload
        if self.offload == "disk":
            self.disk_queue.join()

        # Remove hooks
        self.hook_manager.remove_hooks()

        for layer_id in tqdm(range(len(self.layer_names)), desc="Processing layers"):

            # Skip if no test gradients for this layer
            if not per_layer_test_gradients[layer_id]:
                continue

            # Skip if no train gradients for this layer
            if ifvp_train[layer_id] is None:
                continue

            # Load train gradients based on offload strategy
            if self.offload in ["none", "cpu"]:
                # Load entire train gradients
                train_grads = ifvp_train[layer_id]
                if self.offload == "cpu":
                    train_grads = train_grads.to(self.device)
            elif self.offload == "disk":
                # Load entire file from disk
                train_filename = ifvp_train[layer_id]
                train_grads = self._load_tensor_from_disk(train_filename).to(self.device)

            # Process test batches
            for test_batch_idx in range(len(per_layer_test_gradients[layer_id])):
                # Load test gradient
                if self.offload in ["none", "cpu"]:
                    test_grad = per_layer_test_gradients[layer_id][test_batch_idx]
                    if self.offload == "cpu":
                        test_grad = test_grad.to(self.device)
                elif self.offload == "disk":
                    test_filename = per_layer_test_gradients[layer_id][test_batch_idx]
                    test_grad = self._load_tensor_from_disk(test_filename).to(self.device)

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
        if self.profile:
            return (IF_score, self.profiling_stats)
        else:
            return IF_score