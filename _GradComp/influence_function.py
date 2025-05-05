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


class DiskIOManager:
    """
    Manager for disk I/O operations with thread pool for parallel processing.
    Handles standardized paths, async reading/writing, and metadata management.

    File organization:
    cache_dir/
      ├── grad/
      │   ├── batch_0.pt  # Contains all layer gradients for batch 0
      │   ├── batch_1.pt
      │   └── ...
      ├── ifvp/
      │   ├── batch_0.pt  # Contains all layer IFVP for batch 0
      │   ├── batch_1.pt
      │   └── ...
      └── precond/
          ├── layer_0.pt  # One preconditioner per layer
          ├── layer_1.pt
          └── ...
    """

    def __init__(self, cache_dir: str, setting: str, num_threads: int = 32, hessian: HessianOptions = "raw"):
        """
        Initialize the DiskIOManager.

        Args:
            cache_dir: Directory to save files
            setting: Experiment setting/name
            num_threads: Number of worker threads for I/O operations
            hessian: Hessian approximation type for path generation
        """
        self.cache_dir = cache_dir
        self.setting = setting
        self.num_threads = num_threads
        self.hessian = hessian

        # Create cache directory if it doesn't exist
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.futures = []

    def get_path(self,
                 data_type: DataTypeOptions,
                 batch_idx: Optional[int] = None,
                 layer_idx: Optional[int] = None,
                 is_test: bool = False) -> str:
        """
        Generate standardized path for data storage.

        Args:
            data_type: Type of data ('gradients', 'preconditioners', or 'ifvp')
            batch_idx: Optional batch index for batch-specific files
            layer_idx: Optional layer index for preconditioners
            is_test: Whether this is for test data

        Returns:
            Full path to the file
        """
        if not self.cache_dir:
            raise ValueError("Cache directory is not set")

        # Determine subdirectory based on data type
        if data_type == 'gradients':
            subdir = 'grad'
        elif data_type == 'preconditioners':
            subdir = 'precond'
        elif data_type == 'ifvp':
            subdir = 'ifvp'
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        # Create subdirectory if it doesn't exist
        subdir_path = os.path.join(self.cache_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)

        # Determine filename
        if data_type == 'preconditioners':
            if layer_idx is None:
                raise ValueError("Layer index must be provided for preconditioners")
            filename = f"layer_{layer_idx}.pt"
        else:
            if batch_idx is None:
                raise ValueError("Batch index must be provided for gradients and IFVP")
            prefix = "test_" if is_test else ""
            filename = f"{prefix}batch_{batch_idx}.pt"

        return os.path.join(subdir_path, filename)

    def save_tensor(self, tensor: torch.Tensor, path: str, async_save: bool = True) -> None:
        """
        Save a tensor to disk, optionally asynchronously.
        Uses CUDA stream for efficient CPU transfer and thread pool for disk I/O.

        Args:
            tensor: The tensor to save
            path: Path where to save the tensor
            async_save: Whether to save asynchronously
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Use async stream for CPU transfer if tensor is on CUDA
        if tensor.is_cuda:
            with async_stream():
                cpu_tensor = tensor.cpu()

                if async_save:
                    # Save asynchronously using thread pool
                    future = self.executor.submit(torch.save, cpu_tensor, path)
                    self.futures.append(future)
                else:
                    # Save synchronously
                    torch.save(cpu_tensor, path)
        else:
            # Tensor already on CPU
            if async_save:
                future = self.executor.submit(torch.save, tensor, path)
                self.futures.append(future)
            else:
                torch.save(tensor, path)

    def save_dict(self, data_dict: Dict, path: str, async_save: bool = True) -> None:
        """
        Save a dictionary of tensors to disk, optionally asynchronously.
        Uses CUDA stream for efficient CPU transfer and thread pool for disk I/O.

        Args:
            data_dict: Dictionary of tensors to save
            path: Path where to save the dictionary
            async_save: Whether to save asynchronously
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Process CUDA tensors if present
        has_cuda = any(isinstance(v, torch.Tensor) and v.is_cuda for v in data_dict.values())

        if has_cuda:
            with async_stream():
                # Transfer all tensors to CPU within the stream
                cpu_dict = {k: v.cpu() if isinstance(v, torch.Tensor) and v.is_cuda else v
                          for k, v in data_dict.items()}

                if async_save:
                    # Save asynchronously using thread pool
                    future = self.executor.submit(torch.save, cpu_dict, path)
                    self.futures.append(future)
                else:
                    # Save synchronously
                    torch.save(cpu_dict, path)
        else:
            # No CUDA tensors
            if async_save:
                future = self.executor.submit(torch.save, data_dict, path)
                self.futures.append(future)
            else:
                torch.save(data_dict, path)

    def load_tensor(self, path: str) -> torch.Tensor:
        """
        Load a tensor from disk.

        Args:
            path: Path to the tensor file

        Returns:
            The loaded tensor
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find tensor file: {path}")

        return torch.load(path)

    def load_dict(self, path: str) -> Dict:
        """
        Load a dictionary of tensors from disk.

        Args:
            path: Path to the dictionary file

        Returns:
            The loaded dictionary
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find dictionary file: {path}")

        return torch.load(path)

    def batch_load_tensors(self, paths: List[str], process_fn=None) -> List[torch.Tensor]:
        """
        Load multiple tensors in parallel.

        Args:
            paths: List of paths to tensor files
            process_fn: Optional function to process each tensor after loading

        Returns:
            List of loaded tensors
        """
        # Submit all load jobs to thread pool
        futures = [self.executor.submit(self.load_tensor, path) for path in paths]

        # Wait for all to complete and gather results
        tensors = []
        for future in futures:
            tensor = future.result()
            if process_fn:
                tensor = process_fn(tensor)
            tensors.append(tensor)

        return tensors

    def batch_load_dicts(self, paths: List[str]) -> List[Dict]:
        """
        Load multiple dictionaries in parallel.

        Args:
            paths: List of paths to dictionary files

        Returns:
            List of loaded dictionaries
        """
        # Submit all load jobs to thread pool
        futures = [self.executor.submit(self.load_dict, path) for path in paths]

        # Wait for all to complete and gather results
        return [future.result() for future in futures]

    def find_batch_files(self,
                       data_type: DataTypeOptions,
                       is_test: bool = False) -> List[str]:
        """
        Find all batch files for a specific data type.

        Args:
            data_type: Type of data
            is_test: Whether to look for test data

        Returns:
            List of file paths
        """
        if not self.cache_dir:
            return []

        # Determine subdirectory based on data type
        if data_type == 'gradients':
            subdir = 'grad'
        elif data_type == 'ifvp':
            subdir = 'ifvp'
        else:
            raise ValueError(f"Cannot find batch files for data type: {data_type}")

        # Construct path pattern
        prefix = "test_" if is_test else ""
        pattern = os.path.join(self.cache_dir, subdir, f"{prefix}batch_*.pt")

        # Find all matching files
        return sorted(glob.glob(pattern))

    def extract_batch_idx(self, path: str) -> int:
        """
        Extract batch index from file path.

        Args:
            path: File path

        Returns:
            Batch index
        """
        filename = os.path.basename(path)
        # Extract batch index from filename (e.g., "batch_42.pt" -> 42)
        try:
            return int(filename.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            raise ValueError(f"Could not extract batch index from {filename}")

    def wait_for_async_operations(self) -> None:
        """Wait for all pending async operations to complete."""
        for future in self.futures:
            future.result()
        self.futures = []


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
        self.total_samples = 0

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
                    # Convert string keys back to integers for batch info
                    self.batch_info = {
                        int(batch_idx): {
                            "sample_count": info["sample_count"],
                            "start_idx": info["start_idx"]
                        }
                        for batch_idx, info in metadata['batch_info'].items()
                    }
                    self.total_samples = metadata.get('total_samples', 0)

                print(f"Loaded metadata for {len(self.batch_info)} batches from {metadata_path}")
            except Exception as e:
                print(f"Error loading metadata: {e}")

    def add_batch_info(self, batch_idx: int, sample_count: int) -> None:
        """
        Add information about a processed batch without assuming sample order.
        Only stores the sample count; start indices are computed when saving metadata.
        """
        # Only store the sample count; don't calculate start_idx yet
        self.batch_info[batch_idx] = {
            "sample_count": sample_count,
            "start_idx": 0  # Placeholder, will be properly set when saving
        }

    def save_metadata(self) -> None:
        """Save all metadata to disk with file locking, supporting out-of-order batch processing."""
        if not self.cache_dir:
            return

        metadata_path = self._get_metadata_path()
        lock_path = metadata_path + ".lock"

        # Try to acquire a lock
        try:
            # Simple file-based lock
            with open(lock_path, 'x') as lock_file:
                # First, load any existing metadata to merge with our updates
                existing_batch_info = {}
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            # Convert string keys back to integers
                            existing_batch_info = {
                                int(batch_idx): info
                                for batch_idx, info in metadata.get('batch_info', {}).items()
                            }
                    except Exception as e:
                        print(f"Error loading existing metadata: {e}")

                # Merge our batch info with existing batch info
                merged_batch_info = {**existing_batch_info, **self.batch_info}

                # Recompute all start indices based on batch order
                sorted_batches = sorted(merged_batch_info.keys())
                current_idx = 0

                for batch_idx in sorted_batches:
                    merged_batch_info[batch_idx]["start_idx"] = current_idx
                    current_idx += merged_batch_info[batch_idx]["sample_count"]

                # Calculate total samples from the merged data
                total_samples = current_idx

                # Prepare serializable format
                serializable_info = {
                    str(idx): info for idx, info in merged_batch_info.items()
                }

                metadata = {
                    'batch_info': serializable_info,
                    'layer_names': self.layer_names,
                    'total_samples': total_samples
                }

                # Write to temporary file then rename (atomic operation)
                temp_path = metadata_path + ".tmp"
                with open(temp_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                os.replace(temp_path, metadata_path)

                # Update our in-memory state to match what we saved
                self.batch_info = merged_batch_info
                self.total_samples = total_samples

                print(f"Saved metadata for {len(self.batch_info)} batches to {metadata_path}")
        except FileExistsError:
            # Lock already exists, wait and retry
            print("Metadata is being updated by another process, waiting...")
            time.sleep(1)
            self.save_metadata()  # Recursive retry
        finally:
            # Remove lock file if it exists and we created it
            if os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                except:
                    pass

    def get_batch_indices(self) -> List[int]:
        """Get all batch indices."""
        return sorted(self.batch_info.keys())

    def get_total_samples(self) -> int:
        """Get total number of samples across all batches."""
        return self.total_samples

    def get_batch_to_sample_mapping(self) -> Dict[int, Tuple[int, int]]:
        """
        Get mapping from batch indices to sample ranges.

        Returns:
            Dictionary mapping batch index to (start_sample, end_sample) tuple
        """
        mapping = {}

        for batch_idx in sorted(self.batch_info.keys()):
            info = self.batch_info[batch_idx]
            start_idx = info["start_idx"]
            end_idx = start_idx + info["sample_count"]
            mapping[batch_idx] = (start_idx, end_idx)

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
        batch_range: Optional[Tuple[int, int]] = None
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Cache raw projected gradients from training data to disk or memory.
        Organizes gradients by batch, with each batch file containing all layer gradients.

        Args:
            train_dataloader: DataLoader for the training data
            batch_range: Optional tuple of (start_batch, end_batch) to process only a subset of batches

        Returns:
            Dictionary mapping batch indices to lists of tensors (one tensor per layer)
        """
        # Set default batch range if not provided
        if batch_range is None:
            batch_range = (0, len(train_dataloader))

        # Handle batch range
        start_batch, end_batch = batch_range
        batch_msg = f" (processing batches {start_batch} to {end_batch-1})"

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

        # Create minimal in-memory metadata for non-disk offloading strategies
        if self.offload != "disk" and (not hasattr(self, 'metadata') or self.metadata is None):
            # Create simple metadata object
            self.metadata = type('', (), {})()
            self.metadata.batch_info = {}
            self.metadata.total_samples = 0

            # Add required methods
            self.metadata.get_total_samples = lambda: self.metadata.total_samples
            self.metadata.get_batch_to_sample_mapping = lambda: {
                batch_idx: (info["start_idx"], info["start_idx"] + info["sample_count"])
                for batch_idx, info in self.metadata.batch_info.items()
            }
            self.metadata._load_metadata_if_exists = lambda: None

        # Store metadata about processed batches
        if hasattr(self, 'metadata'):
            # Now add batch info for each processed batch
            start_batch, _ = batch_range
            current_total = getattr(self.metadata, 'total_samples', 0)

            for i, batch_idx in enumerate(sorted(gradients_dict.keys())):
                # Find the corresponding sample count
                sample_idx = batch_idx - start_batch
                if 0 <= sample_idx < len(batch_sample_counts):
                    sample_count = batch_sample_counts[sample_idx]

                    # Add this batch to metadata
                    if self.offload == "disk":
                        self.metadata.add_batch_info(batch_idx=batch_idx, sample_count=sample_count)
                    else:
                        # For non-disk offloading, update in-memory metadata
                        self.metadata.batch_info[batch_idx] = {
                            "sample_count": sample_count,
                            "start_idx": current_total
                        }
                        current_total += sample_count

            # Update total samples
            if self.offload != "disk":
                self.metadata.total_samples = current_total

            # Save the updated metadata if using disk
            if self.offload == "disk":
                self.metadata.save_metadata()

        # Store gradients in memory if not using disk offload
        if self.offload != "disk":
            if not hasattr(self, 'cached_gradients') or self.cached_gradients is None:
                self.cached_gradients = {}
            # Update cached gradients with new batch data
            self.cached_gradients.update(gradients_dict)

        print(f"Cached gradients for {len(self.layer_names)} modules across {len(gradients_dict)} batches")

        # Remove hooks after collecting all gradients
        if self.hook_manager:
            self.hook_manager.remove_hooks()
            self.hook_manager = None

        if self.profile and self.profiling_stats:
            return (gradients_dict, self.profiling_stats)
        else:
            return gradients_dict

    def compute_preconditioners(self, damping: Optional[float] = None, save: bool = False) -> List[torch.Tensor]:
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

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Initialize Hessian accumulators for all layers
        hessian_accumulators = [None] * len(self.layer_names)
        sample_counts = [0] * len(self.layer_names)

        if self.offload == "disk":
            batch_files = self.disk_io.find_batch_files('gradients')

            # Process in chunks to avoid memory issues
            chunk_size = 1  # Adjust based on memory constraints
            for i in tqdm(range(0, len(batch_files), chunk_size), desc="Processing batches for preconditioners..."):
                chunk_files = batch_files[i:i+chunk_size]

                # Load gradient dictionaries in parallel
                batch_dicts = self.disk_io.batch_load_dicts(chunk_files)

                # For each layer, batch process all batch dicts
                for layer_idx in range(len(self.layer_names)):
                    # Collect gradients from all batches in this chunk for the current layer
                    gradients_list = []
                    for batch_dict in batch_dicts:
                        if layer_idx in batch_dict and batch_dict[layer_idx].numel() > 0:
                            gradients_list.append(batch_dict[layer_idx])

                    if not gradients_list:
                        continue  # Skip if no gradients for this layer in this chunk

                    # Combine all gradients into a single tensor (true batching)
                    combined_gradients = torch.cat(gradients_list, dim=0).to(self.device)

                    # Single matrix multiplication for all batches
                    batch_hessian = torch.matmul(combined_gradients.t(), combined_gradients)

                    # Update accumulator
                    if hessian_accumulators[layer_idx] is None:
                        hessian_accumulators[layer_idx] = batch_hessian
                    else:
                        hessian_accumulators[layer_idx] += batch_hessian

                    # Update sample count
                    sample_counts[layer_idx] += combined_gradients.shape[0]

                    # Clean up
                    del combined_gradients, batch_hessian
                    torch.cuda.empty_cache()

                # Clean up batch dictionaries
                del batch_dicts
                torch.cuda.empty_cache()
        else:
            # For in-memory processing, iterate through batch indices
            if self.cached_gradients:
                for batch_idx, batch_grads in tqdm(self.cached_gradients.items(), desc="Processing batches for preconditioners..."):
                    # Process each layer in this batch
                    for layer_idx in range(len(self.layer_names)):
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
                        if hessian_accumulators[layer_idx] is None:
                            hessian_accumulators[layer_idx] = batch_hessian
                        else:
                            hessian_accumulators[layer_idx] += batch_hessian

                        # Update sample count
                        sample_counts[layer_idx] += grad.shape[0]

                        # Clean up to save memory if needed
                        if self.offload == "cpu":
                            del grad, batch_hessian
                            torch.cuda.empty_cache()

        # Compute preconditioners from accumulated Hessians
        preconditioners = [None] * len(self.layer_names)

        # Process each layer's accumulated Hessian
        for layer_idx in tqdm(range(len(self.layer_names)), desc="Computing preconditioners..."):
            hessian_accumulator = hessian_accumulators[layer_idx]
            sample_count = sample_counts[layer_idx]

            # If we have accumulated Hessian, compute preconditioner
            if hessian_accumulator is not None and sample_count > 0:
                # Normalize by total number of samples
                hessian = hessian_accumulator / sample_count

                # Compute inverse based on Hessian type
                if self.hessian == "raw":
                    precond = stable_inverse(hessian, damping=damping)

                    # Save or store based on offload strategy
                    if self.offload == "disk" and save:
                        cpu_precond = precond.cpu()
                        file_path = self.disk_io.get_path('preconditioners', layer_idx=layer_idx)
                        self.disk_io.save_tensor(cpu_precond, file_path)
                        preconditioners[layer_idx] = cpu_precond
                    else:
                        # Store in memory
                        if self.cpu_offload:
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

        Optimized to process multiple batches at once and reduce device transfers.

        Args:
            save: Whether to save the results to disk (for disk offload)

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
                # Process gradient files in parallel when possible
                ifvp_dict = {}
                batch_files = self.disk_io.find_batch_files('gradients')

                # Create a ThreadPoolExecutor for parallel processing
                with ThreadPoolExecutor(max_workers=min(32, len(batch_files))) as executor:
                    # Submit all copy tasks to the executor
                    futures = []
                    for file_path in batch_files:
                        batch_idx = self.disk_io.extract_batch_idx(file_path)
                        ifvp_path = self.disk_io.get_path('ifvp', batch_idx=batch_idx)
                        # Copy file or load and save
                        futures.append(executor.submit(self._copy_grad_to_ifvp, file_path, ifvp_path, batch_idx))

                    # Collect results
                    for future in futures:
                        batch_idx, batch_data = future.result()
                        ifvp_dict[batch_idx] = batch_data

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

            # Process in chunks to leverage batch operations
            chunk_size = 1  # Adjust based on memory and GPU capability

            for i in tqdm(range(0, len(batch_files), chunk_size), desc="Processing batches for ifvp..."):
                chunk_files = batch_files[i:i+chunk_size]
                chunk_batch_indices = [self.disk_io.extract_batch_idx(path) for path in chunk_files]

                # Load all gradient dictionaries in parallel
                batch_grad_dicts = self.disk_io.batch_load_dicts(chunk_files)

                # Process each layer across all batches in the chunk
                for layer_idx, precond in enumerate(preconditioners):
                    # Skip if preconditioner is None
                    if precond is None:
                        continue

                    # Load preconditioner to device if needed
                    if isinstance(precond, str):
                        precond_tensor = self.disk_io.load_tensor(precond).to(self.device)
                    elif isinstance(precond, list):
                        print(f"Warning: Preconditioner for layer {layer_idx} is a list, not a tensor or file path.")
                        continue
                    else:
                        precond_tensor = precond.to(self.device)

                    # Process all batches for this layer
                    for batch_dict, batch_idx in zip(batch_grad_dicts, chunk_batch_indices):
                        # Skip if layer not in gradient dictionary
                        if layer_idx not in batch_dict or batch_dict[layer_idx].numel() == 0:
                            # Create ifvp dict entry if it doesn't exist
                            if batch_idx not in ifvp_dict:
                                ifvp_dict[batch_idx] = {}
                            # Add empty tensor for this layer
                            ifvp_dict[batch_idx][layer_idx] = torch.tensor([])
                            continue

                        # Load gradient to device
                        grad = batch_dict[layer_idx].to(self.device)

                        # Compute IFVP
                        result = torch.matmul(precond_tensor, grad.t()).t()

                        # Create ifvp dict entry if it doesn't exist
                        if batch_idx not in ifvp_dict:
                            ifvp_dict[batch_idx] = {}

                        # Store in dictionary
                        ifvp_dict[batch_idx][layer_idx] = result.cpu()

                        # Clean up
                        del grad, result

                    # Clean up preconditioner
                    del precond_tensor
                    torch.cuda.empty_cache()

                # Save all IFVP batches to disk
                save_tasks = []
                for batch_idx, batch_dict in ifvp_dict.items():
                    if batch_idx in chunk_batch_indices:  # Only save batches from current chunk
                        ifvp_path = self.disk_io.get_path('ifvp', batch_idx=batch_idx)
                        save_tasks.append((batch_dict, ifvp_path))

                # Save in parallel
                with ThreadPoolExecutor(max_workers=min(16, len(save_tasks))) as executor:
                    for batch_dict, ifvp_path in save_tasks:
                        executor.submit(self.disk_io.save_dict, batch_dict, ifvp_path)

                # Convert dictionary format to list format for consistency
                for batch_idx in list(ifvp_dict.keys()):
                    if batch_idx in chunk_batch_indices:  # Only process batches from current chunk
                        batch_dict = ifvp_dict[batch_idx]
                        ifvp_dict[batch_idx] = [batch_dict.get(i, torch.tensor([]))
                                            for i in range(len(self.layer_names))]

                # Clean up batch dictionaries
                del batch_grad_dicts
                torch.cuda.empty_cache()
        else:
            # For in-memory processing, we can optimize by processing layers in parallel
            if self.cached_gradients:
                # Get all batch indices
                batch_indices = sorted(self.cached_gradients.keys())

                # Process each layer
                for layer_idx, precond in enumerate(preconditioners):
                    # Skip if preconditioner is None
                    if precond is None:
                        continue

                    # Move preconditioner to device
                    if isinstance(precond, str):
                        precond_tensor = torch.load(precond).to(self.device)
                    elif isinstance(precond, list):
                        print(f"Warning: Preconditioner for layer {layer_idx} is a list, not a tensor or file path.")
                        continue
                    else:
                        precond_tensor = precond.to(self.device) if self.offload == "cpu" else precond

                    # Process batches in chunks to manage memory
                    chunk_size = 32
                    for i in range(0, len(batch_indices), chunk_size):
                        chunk_batch_indices = batch_indices[i:i+chunk_size]

                        # Process each batch in the chunk
                        for batch_idx in chunk_batch_indices:
                            batch_grads = self.cached_gradients[batch_idx]

                            # Skip if layer_idx is out of range or gradient is empty
                            if layer_idx >= len(batch_grads) or batch_grads[layer_idx].numel() == 0:
                                # Initialize ifvp list if not exists
                                if batch_idx not in ifvp_dict:
                                    ifvp_dict[batch_idx] = [torch.tensor([]) for _ in range(len(self.layer_names))]
                                # Skip to next batch
                                continue

                            # Get gradient
                            grad = batch_grads[layer_idx]

                            # Move to device if needed
                            if self.offload == "cpu":
                                grad = grad.to(self.device)

                            # Compute IFVP
                            result = torch.matmul(precond_tensor, grad.t()).t()

                            # Initialize ifvp list if not exists
                            if batch_idx not in ifvp_dict:
                                ifvp_dict[batch_idx] = [torch.tensor([]) for _ in range(len(self.layer_names))]

                            # Store result
                            if self.offload == "cpu":
                                ifvp_dict[batch_idx][layer_idx] = result.cpu()
                                del result
                            else:
                                ifvp_dict[batch_idx][layer_idx] = result

                            # Clean up
                            if self.offload == "cpu":
                                del grad

                        # Clear cache after processing a chunk
                        if self.offload == "cpu":
                            torch.cuda.empty_cache()

                    # Clean up preconditioner
                    del precond_tensor
                    torch.cuda.empty_cache()

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

    def _copy_grad_to_ifvp(self, grad_path, ifvp_path, batch_idx):
        """Helper method to copy gradient files to IFVP files in parallel"""
        # Load gradient dictionary
        grad_dict = self.disk_io.load_dict(grad_path)

        # Save as IFVP
        self.disk_io.save_dict(grad_dict, ifvp_path)

        # Convert to list format for consistency
        batch_data = [grad_dict[i] if i in grad_dict else torch.tensor([])
                    for i in range(len(self.layer_names))]

        return batch_idx, batch_data

    def attribute(
        self,
        test_dataloader: DataLoader,
        train_dataloader: Optional[DataLoader] = None,
        use_cached_ifvp: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ProfilingStats]]:
        """
        Attribute influence of training examples on test examples.
        Optimized to process multiple batches simultaneously and reduce device transfers.

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

        # Get all train batch indices and organize into optimal chunks
        train_batch_indices = sorted(batch_to_sample_mapping.keys())

        # Determine optimal chunk sizes based on memory and dataset characteristics
        train_chunk_size = min(1, len(train_batch_indices))  # Adjust based on memory available
        test_chunk_size = min(1, len(test_batch_indices))     # Process multiple test batches at once

        # Calculate influence scores by processing chunks of train and test batches
        for test_chunk_start in tqdm(range(0, len(test_grads_dict), test_chunk_size), desc="Attributing test batches..."):
            test_chunk_end = min(test_chunk_start + test_chunk_size, len(test_grads_dict))
            test_chunk_indices = sorted(list(test_grads_dict.keys()))[test_chunk_start:test_chunk_end]

            # Load all test gradients for this chunk
            test_chunk_grads = {}
            for test_batch_idx in test_chunk_indices:
                if self.offload == "disk":
                    test_path = self.disk_io.get_path('gradients', batch_idx=test_batch_idx, is_test=True)
                    test_dict = self.disk_io.load_dict(test_path)
                    test_chunk_grads[test_batch_idx] = [test_dict[i] if i in test_dict else torch.tensor([])
                                                    for i in range(len(self.layer_names))]
                else:
                    test_chunk_grads[test_batch_idx] = test_grads_dict[test_batch_idx]

            # Process train batches in chunks
            for train_chunk_start in range(0, len(train_batch_indices), train_chunk_size):
                train_chunk_end = min(train_chunk_start + train_chunk_size, len(train_batch_indices))
                train_chunk_indices = train_batch_indices[train_chunk_start:train_chunk_end]

                # Load all IFVP for this train chunk
                train_chunk_ifvps = {}
                for train_batch_idx in train_chunk_indices:
                    if self.offload == "disk":
                        ifvp_path = self.disk_io.get_path('ifvp', batch_idx=train_batch_idx)
                        ifvp_dict_batch = self.disk_io.load_dict(ifvp_path)
                        train_chunk_ifvps[train_batch_idx] = [ifvp_dict_batch[i] if i in ifvp_dict_batch else torch.tensor([])
                                                            for i in range(len(self.layer_names))]
                    else:
                        train_chunk_ifvps[train_batch_idx] = ifvp_dict[train_batch_idx]

                # Process each layer across all batches in the chunks
                for layer_idx in range(len(self.layer_names)):
                    # Skip if layer has no data
                    has_data = any(train_ifvps[layer_idx].numel() > 0 for train_ifvps in train_chunk_ifvps.values())
                    has_data = has_data and any(test_grads[layer_idx].numel() > 0 for test_grads in test_chunk_grads.values())
                    if not has_data:
                        continue

                    # Process each test batch with this layer
                    for test_batch_idx in test_chunk_indices:
                        test_grads = test_chunk_grads[test_batch_idx][layer_idx]
                        if test_grads.numel() == 0:
                            continue

                        # Get column indices for this test batch
                        col_st, col_ed = test_batch_indices[test_batch_idx]

                        # Move test gradients to device
                        test_grads_gpu = test_grads.to(self.device)

                        # Process each train batch for this test batch and layer
                        for train_batch_idx in train_chunk_indices:
                            train_ifvp = train_chunk_ifvps[train_batch_idx][layer_idx]
                            if train_ifvp.numel() == 0:
                                continue

                            # Get row indices for this train batch
                            row_st, row_ed = batch_to_sample_mapping[train_batch_idx]

                            # Move IFVP to device
                            train_ifvp_gpu = train_ifvp.to(self.device)

                            # Compute influence for this layer, train batch, and test batch
                            with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision for faster computation
                                layer_influence = torch.matmul(train_ifvp_gpu, test_grads_gpu.t())

                            # Update influence scores
                            IF_score[row_st:row_ed, col_st:col_ed] += layer_influence.cpu()

                            # Clean up GPU memory
                            del train_ifvp_gpu, layer_influence

                        # Clean up test gradients
                        del test_grads_gpu

                    # Clear cache after processing each layer
                    torch.cuda.empty_cache()

                # Clean up train chunk
                del train_chunk_ifvps
                torch.cuda.empty_cache()

            # Clean up test chunk
            del test_chunk_grads
            torch.cuda.empty_cache()

        # print(IF_score)
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

    def _get_preconditioners(self) -> List[Union[torch.Tensor, str, None]]:
        """
        Get preconditioners - either from memory or disk.
        Computes them if not already available.

        Returns:
            List of preconditioners for each layer (tensors, file paths, or None)
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
        result = self.compute_preconditioners(save=False)
        if self.profile:
            return result[0]
        else:
            return result

    def __del__(self) -> None:
        """
        Clean up resources when the object is garbage collected.
        """
        # Clean up memory
        if hasattr(self, 'hook_manager') and self.hook_manager is not None:
            self.hook_manager.remove_hooks()
            self.hook_manager = None