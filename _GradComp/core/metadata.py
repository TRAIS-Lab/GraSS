"""
Metadata management with worker coordination support.
"""

import os
import json
import time
import threading
import fcntl
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import logging
logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """Complete dataset information computed upfront."""
    total_batches: int
    total_samples: int
    batch_size: int
    batch_to_sample_mapping: Dict[int, Tuple[int, int]]  # batch_idx -> (start_sample, end_sample)

class MetadataManager:
    """
    Metadata manager with worker coordination support.
    Supports pre-computing full dataset info and coordinated partial updates.
    """

    def __init__(self, cache_dir: str, layer_names: List[str]):
        self.cache_dir = cache_dir
        self.layer_names = layer_names
        self.batch_info = {}  # Maps batch_idx -> {sample_count, start_idx, worker_info}
        self.total_samples = 0
        self.layer_dims = None
        self.total_proj_dim = None

        # Full dataset information (computed once, shared by all workers)
        self.dataset_info: Optional[DatasetInfo] = None

        self._metadata_lock = threading.Lock()
        self._pending_batches = {}  # Buffer for batches before bulk save
        self._last_save_time = 0
        self._save_interval = 5.0  # Save every 5 seconds max

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_metadata_if_exists()

        logger.debug(f"Initialized MetadataManager with {len(layer_names)} layers")

    def initialize_full_dataset(self, train_dataloader) -> None:
        """
        Initialize full dataset metadata from dataloader.
        Thread-safe - can be called by multiple workers simultaneously.
        """
        if self.dataset_info is not None:
            logger.debug("Dataset info already initialized")
            return

        # Use file locking to ensure only one worker initializes
        lock_path = os.path.join(self.cache_dir, ".metadata_init.lock") if self.cache_dir else None

        if lock_path:
            with open(lock_path, 'w') as lock_file:
                try:
                    # Try to acquire exclusive lock (non-blocking first to check)
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    logger.info("Acquired metadata initialization lock - initializing dataset...")

                    # Double-check if another worker initialized while we were waiting
                    self._load_metadata_if_exists()
                    if self.dataset_info is not None:
                        logger.info("Dataset already initialized by another worker")
                        return

                    # Proceed with initialization
                    self._perform_dataset_initialization(train_dataloader)

                except BlockingIOError:
                    # Another worker is initializing, wait for it
                    logger.info("Another worker is initializing dataset metadata, waiting...")
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)  # Wait for exclusive lock

                    # Load the metadata that the other worker created
                    self._load_metadata_if_exists()
                    if self.dataset_info is None:
                        logger.error("Dataset initialization failed by other worker")
                        raise RuntimeError("Dataset metadata initialization failed")

                    logger.info("Dataset metadata initialization completed by another worker")
        else:
            # No cache directory, initialize directly (single-process mode)
            self._perform_dataset_initialization(train_dataloader)

    def _perform_dataset_initialization(self, train_dataloader) -> None:
        """Perform the actual dataset initialization."""
        total_batches = len(train_dataloader)
        batch_size = train_dataloader.batch_size

        logger.info("Computing full dataset information...")

        batch_to_sample_mapping = {}
        current_sample_idx = 0

        # Compute accurate sample counts
        if hasattr(train_dataloader.dataset, '__len__'):
            dataset_size = len(train_dataloader.dataset)

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                # Handle last batch which might be smaller
                if batch_idx == total_batches - 1 and not train_dataloader.drop_last:
                    actual_batch_size = dataset_size - start_idx
                else:
                    actual_batch_size = min(batch_size, dataset_size - start_idx)

                batch_to_sample_mapping[batch_idx] = (current_sample_idx, current_sample_idx + actual_batch_size)
                current_sample_idx += actual_batch_size
        else:
            # Fallback: assume uniform batch sizes
            logger.warning("Dataset doesn't support __len__, using fallback batch size computation")
            for batch_idx in range(total_batches):
                batch_to_sample_mapping[batch_idx] = (current_sample_idx, current_sample_idx + batch_size)
                current_sample_idx += batch_size

        # Create dataset info
        self.dataset_info = DatasetInfo(
            total_batches=total_batches,
            total_samples=current_sample_idx,
            batch_size=batch_size,
            batch_to_sample_mapping=batch_to_sample_mapping
        )

        # Initialize batch_info with the full structure
        for batch_idx, (start_sample, end_sample) in batch_to_sample_mapping.items():
            self.batch_info[batch_idx] = {
                "sample_count": end_sample - start_sample,
                "start_idx": start_sample
            }

        self.total_samples = current_sample_idx

        # Save the complete dataset metadata immediately
        self._save_dataset_metadata()

        logger.info(f"Initialized full dataset: {total_batches} batches, {current_sample_idx} samples")

    def update_worker_batch_info(self, batch_idx: int, sample_count: int, worker_id: str) -> None:
        """
        Update batch info from a specific worker for validation and tracking.
        """
        with self._metadata_lock:
            if batch_idx in self.batch_info:
                # Validate sample count matches expectations
                expected_count = self.batch_info[batch_idx]["sample_count"]
                if sample_count != expected_count:
                    logger.warning(f"Worker {worker_id}: Batch {batch_idx} sample count mismatch. "
                                 f"Expected: {expected_count}, Got: {sample_count}")

                # Add worker-specific tracking info
                self.batch_info[batch_idx][f"worker_{worker_id}"] = {
                    "processed": True,
                    "sample_count": sample_count,
                    "timestamp": time.time()
                }
            else:
                logger.error(f"Worker {worker_id}: Attempting to update unknown batch {batch_idx}")

    def add_batch_info(self, batch_idx: int, sample_count: int) -> None:
        """Backward compatibility method - delegates to worker-specific update."""
        # Extract worker ID from thread context if available
        import threading
        thread_name = threading.current_thread().name
        worker_id = thread_name if "worker" in thread_name.lower() else "unknown"
        self.update_worker_batch_info(batch_idx, sample_count, worker_id)

    def set_layer_dims(self, layer_dims: List[int]) -> None:
        """Set the layer dimensions and update saved metadata."""
        with self._metadata_lock:
            self.layer_dims = layer_dims
            self.total_proj_dim = sum(layer_dims) if layer_dims else None
            logger.debug(f"Set layer dimensions: {len(layer_dims)} layers, total={self.total_proj_dim}")

            # Update saved metadata if we have full dataset info
            if self.dataset_info is not None:
                self._save_dataset_metadata()

    def get_total_samples(self) -> int:
        """Get the total number of samples across all batches."""
        if self.dataset_info:
            return self.dataset_info.total_samples
        return self.total_samples

    def get_batch_to_sample_mapping(self) -> Dict[int, Tuple[int, int]]:
        """Get mapping from batch indices to sample index ranges."""
        if self.dataset_info:
            return self.dataset_info.batch_to_sample_mapping

        # Fallback to computed mapping
        return {
            batch_idx: (info["start_idx"], info["start_idx"] + info["sample_count"])
            for batch_idx, info in self.batch_info.items()
        }

    def get_total_batches(self) -> int:
        """Get total number of batches in the dataset."""
        if self.dataset_info:
            return self.dataset_info.total_batches
        return len(self.batch_info)

    def save_metadata(self) -> None:
        """Save metadata - requires dataset info to be initialized."""
        if not self.cache_dir:
            return

        if self.dataset_info is not None:
            self._save_dataset_metadata()
        else:
            logger.error("Cannot save metadata without dataset initialization")
            raise ValueError("Dataset metadata must be initialized first. Call initialize_full_dataset().")

    def _save_dataset_metadata(self) -> None:
        """Save the complete dataset metadata with file locking."""
        if not self.cache_dir or self.dataset_info is None:
            return

        metadata_path = self._get_metadata_path()
        temp_path = metadata_path + ".tmp"

        try:
            # Flush any pending batch updates
            with self._metadata_lock:
                self._flush_pending_batches()

            # Prepare complete metadata structure
            metadata = {
                'batch_info': {str(k): v for k, v in self.batch_info.items()},
                'layer_names': self.layer_names,
                'layer_dims': self.layer_dims,
                'total_proj_dim': self.total_proj_dim,
                'total_samples': self.total_samples,
                'dataset_info': {
                    'total_batches': self.dataset_info.total_batches,
                    'total_samples': self.dataset_info.total_samples,
                    'batch_size': self.dataset_info.batch_size,
                    'batch_to_sample_mapping': {str(k): v for k, v in self.dataset_info.batch_to_sample_mapping.items()}
                },
                'timestamp': time.time()
            }

            # Atomic write with file locking
            with open(temp_path, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(metadata, f, separators=(',', ':'))
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            os.replace(temp_path, metadata_path)
            logger.debug("Saved complete dataset metadata")

        except Exception as e:
            logger.error(f"Error saving dataset metadata: {e}")
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise

    def _flush_pending_batches(self) -> None:
        """Flush pending batches to the main batch_info dict."""
        if not self._pending_batches:
            return

        self.batch_info.update(self._pending_batches)
        self._pending_batches.clear()
        self._last_save_time = time.time()

    def _get_metadata_path(self) -> str:
        """Get the path to the metadata file."""
        return os.path.join(self.cache_dir, "batch_metadata.json")

    def _load_metadata_if_exists(self) -> None:
        """Load metadata from disk if it exists."""
        metadata_path = self._get_metadata_path()
        if not os.path.exists(metadata_path):
            return

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Load batch info
            self.batch_info = {
                int(batch_idx): info for batch_idx, info in metadata['batch_info'].items()
            }

            self.total_samples = metadata.get('total_samples', 0)

            # Load layer information
            if 'layer_names' in metadata:
                self.layer_names = metadata['layer_names']
            if 'layer_dims' in metadata:
                self.layer_dims = metadata['layer_dims']
                self.total_proj_dim = metadata.get('total_proj_dim')

            # Load full dataset info (required for new format)
            if 'dataset_info' in metadata:
                dataset_info_dict = metadata['dataset_info']
                self.dataset_info = DatasetInfo(
                    total_batches=dataset_info_dict['total_batches'],
                    total_samples=dataset_info_dict['total_samples'],
                    batch_size=dataset_info_dict['batch_size'],
                    batch_to_sample_mapping={
                        int(k): v for k, v in dataset_info_dict['batch_to_sample_mapping'].items()
                    }
                )
                logger.debug("Loaded complete dataset information from metadata")
            else:
                logger.warning("Metadata file exists but lacks dataset_info. "
                             "Please reinitialize with initialize_full_dataset().")

            logger.info(f"Loaded metadata for {len(self.batch_info)} batches")

        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            logger.warning("Consider deleting corrupted metadata file and reinitializing")

    def __del__(self):
        """Ensure metadata is saved on destruction."""
        try:
            if hasattr(self, '_pending_batches') and self._pending_batches:
                logger.info("Saving pending metadata on destruction")
                self.save_metadata()
        except Exception as e:
            logger.error(f"Error saving metadata during cleanup: {e}")