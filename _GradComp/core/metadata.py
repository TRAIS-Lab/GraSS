"""
Metadata management with atomic merging for parallel workers.
"""

import os
import json
import time
import threading
from typing import Dict, List, Tuple

import logging
logger = logging.getLogger(__name__)

class MetadataManager:
    """
    Manager for batch metadata with atomic merging support for parallel workers.
    """

    def __init__(self, cache_dir: str, layer_names: List[str]):
        """
        Initialize the metadata manager.

        Args:
            cache_dir: Directory for metadata files
            layer_names: Names of model layers
        """
        self.cache_dir = cache_dir
        self.layer_names = layer_names
        self.batch_info = {}  # Maps batch_idx -> {sample_count, start_idx}
        self.total_samples = 0
        self.layer_dims = None  # Store layer dimensions
        self.total_proj_dim = None  # Total projection dimension
        self._metadata_lock = threading.Lock()
        self._pending_batches = {}  # Buffer for batches before bulk save
        self._last_save_time = 0
        self._save_interval = 5.0  # Save every 5 seconds max

        # Pre-computed dataset information (for full dataset structure)
        self._dataset_total_batches = None
        self._dataset_batch_size = None

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_metadata_if_exists()

        logger.debug(f"Initialized MetadataManager with {len(layer_names)} layers")

    def initialize_dataset_structure(self, train_dataloader) -> None:
        """
        Initialize the complete dataset structure from dataloader.
        This is called once to set up the full batch structure.

        Args:
            train_dataloader: The training dataloader
        """
        if self._dataset_total_batches is not None:
            return  # Already initialized

        total_batches = len(train_dataloader)
        batch_size = train_dataloader.batch_size

        logger.info(f"Initializing dataset structure: {total_batches} batches, batch_size={batch_size}")

        # Pre-compute the complete batch structure
        if hasattr(train_dataloader.dataset, '__len__'):
            dataset_size = len(train_dataloader.dataset)

            # Initialize complete batch info structure
            current_sample_idx = 0
            complete_batch_info = {}

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                # Handle last batch which might be smaller
                if batch_idx == total_batches - 1 and not getattr(train_dataloader, 'drop_last', False):
                    actual_batch_size = dataset_size - start_idx
                else:
                    actual_batch_size = min(batch_size, dataset_size - start_idx)

                complete_batch_info[batch_idx] = {
                    "sample_count": actual_batch_size,
                    "start_idx": current_sample_idx,
                    "processed": False  # Track which batches have been processed
                }
                current_sample_idx += actual_batch_size
        else:
            # Fallback: assume uniform batch sizes
            logger.warning("Dataset doesn't support __len__, using uniform batch size assumption")
            complete_batch_info = {}
            for batch_idx in range(total_batches):
                complete_batch_info[batch_idx] = {
                    "sample_count": batch_size,
                    "start_idx": batch_idx * batch_size,
                    "processed": False
                }
            current_sample_idx = total_batches * batch_size

        # Store the complete structure
        with self._metadata_lock:
            self.batch_info = complete_batch_info
            self.total_samples = current_sample_idx
            self._dataset_total_batches = total_batches
            self._dataset_batch_size = batch_size

        logger.info(f"Initialized complete dataset structure: {total_batches} batches, {current_sample_idx} total samples")

    def add_batch_info(self, batch_idx: int, sample_count: int) -> None:
        """
        Mark a batch as processed and validate sample count.

        Args:
            batch_idx: Index of the batch
            sample_count: Number of samples in the batch (for validation)
        """
        with self._metadata_lock:
            if batch_idx in self.batch_info:
                # Validate sample count matches pre-computed structure
                expected_count = self.batch_info[batch_idx]["sample_count"]
                if sample_count != expected_count:
                    logger.warning(f"Batch {batch_idx} sample count mismatch. "
                                 f"Expected: {expected_count}, Got: {sample_count}")

                # Mark as processed
                self.batch_info[batch_idx]["processed"] = True
                self.batch_info[batch_idx]["processed_time"] = time.time()

                # Add to pending updates for periodic save
                self._pending_batches[batch_idx] = self.batch_info[batch_idx].copy()

                # Check if we should save (time-based or buffer size-based)
                current_time = time.time()
                should_save = (
                    len(self._pending_batches) >= 50 or  # Buffer size threshold
                    (current_time - self._last_save_time) >= self._save_interval  # Time threshold
                )

                if should_save:
                    self._flush_pending_batches()
            else:
                logger.error(f"Attempting to update unknown batch {batch_idx}. "
                           f"Dataset structure may not be initialized.")

    def set_layer_dims(self, layer_dims: List[int]) -> None:
        """
        Set the layer dimensions.

        Args:
            layer_dims: List of dimensions for each layer
        """
        with self._metadata_lock:
            self.layer_dims = layer_dims
            self.total_proj_dim = sum(layer_dims) if layer_dims else None
            logger.debug(f"Set layer dimensions: {len(layer_dims)} layers, total={self.total_proj_dim}")

    def _flush_pending_batches(self) -> None:
        """Flush pending batches to the main batch_info dict."""
        if not self._pending_batches:
            return

        # Merge pending batches (they should already be in batch_info)
        self._pending_batches.clear()
        self._last_save_time = time.time()

    def get_total_samples(self) -> int:
        """Get the total number of samples across all batches."""
        return self.total_samples

    def get_batch_to_sample_mapping(self) -> Dict[int, Tuple[int, int]]:
        """Get mapping from batch indices to sample index ranges."""
        with self._metadata_lock:
            return {
                batch_idx: (info["start_idx"], info["start_idx"] + info["sample_count"])
                for batch_idx, info in self.batch_info.items()
            }

    def get_total_batches(self) -> int:
        """Get total number of batches in the dataset."""
        return len(self.batch_info)

    def save_metadata(self) -> None:
        """
        Save metadata with atomic merging to handle concurrent workers.
        """
        if not self.cache_dir:
            return

        metadata_path = self._get_metadata_path()
        temp_path = metadata_path + f".tmp.{os.getpid()}.{int(time.time() * 1000000)}"

        try:
            with self._metadata_lock:
                # Flush any pending batch updates first
                self._flush_pending_batches()

                # Load existing metadata and merge (atomic read-modify-write)
                existing_metadata = self._load_existing_metadata_safe()

                # Merge batch info: keep the complete structure, update processed status
                merged_batch_info = existing_metadata.get('batch_info', {})

                # Update with our current batch info
                for batch_idx, info in self.batch_info.items():
                    batch_key = str(batch_idx)
                    if batch_key in merged_batch_info:
                        # Preserve existing structure, update processing status
                        merged_batch_info[batch_key].update({
                            "processed": info.get("processed", False),
                            "processed_time": info.get("processed_time", time.time())
                        })
                    else:
                        # Add new batch info
                        merged_batch_info[batch_key] = {
                            "sample_count": info["sample_count"],
                            "start_idx": info["start_idx"],
                            "processed": info.get("processed", False),
                            "processed_time": info.get("processed_time", time.time())
                        }

                # Create complete metadata structure
                metadata = {
                    'batch_info': merged_batch_info,
                    'layer_names': self.layer_names,
                    'layer_dims': self.layer_dims,
                    'total_proj_dim': self.total_proj_dim,
                    'total_samples': self.total_samples,
                    'dataset_total_batches': self._dataset_total_batches,
                    'dataset_batch_size': self._dataset_batch_size,
                    'timestamp': time.time(),
                    'pid': os.getpid()  # Track which process saved this
                }

                # Write to temporary file (atomic operation)
                with open(temp_path, 'w') as f:
                    json.dump(metadata, f, separators=(',', ':'))
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk

                # Atomic rename (this is atomic on most filesystems)
                os.replace(temp_path, metadata_path)

                logger.debug(f"Saved metadata for {len(merged_batch_info)} batches "
                           f"(PID: {os.getpid()})")

        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise

    def _load_existing_metadata_safe(self) -> Dict:
        """Safely load existing metadata with retry logic."""
        metadata_path = self._get_metadata_path()

        for attempt in range(3):  # Retry up to 3 times
            try:
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        return json.load(f)
                else:
                    return {}
            except (json.JSONDecodeError, IOError) as e:
                if attempt < 2:  # Not the last attempt
                    logger.warning(f"Failed to load metadata (attempt {attempt + 1}): {e}. Retrying...")
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed to load metadata after 3 attempts: {e}")
                    return {}

        return {}

    def _get_metadata_path(self) -> str:
        """Get the path to the metadata file."""
        return os.path.join(self.cache_dir, "batch_metadata.json")

    def _load_metadata_if_exists(self) -> None:
        """Load metadata from disk if it exists."""
        existing_metadata = self._load_existing_metadata_safe()

        if not existing_metadata:
            return

        try:
            # Convert string keys back to integers for batch info
            if 'batch_info' in existing_metadata:
                self.batch_info = {
                    int(batch_idx): {
                        "sample_count": info["sample_count"],
                        "start_idx": info["start_idx"],
                        "processed": info.get("processed", False),
                        "processed_time": info.get("processed_time", time.time())
                    }
                    for batch_idx, info in existing_metadata['batch_info'].items()
                }

            self.total_samples = existing_metadata.get('total_samples', 0)
            self._dataset_total_batches = existing_metadata.get('dataset_total_batches')
            self._dataset_batch_size = existing_metadata.get('dataset_batch_size')

            # Load layer information
            if 'layer_names' in existing_metadata:
                self.layer_names = existing_metadata['layer_names']

            if 'layer_dims' in existing_metadata:
                self.layer_dims = existing_metadata['layer_dims']
                self.total_proj_dim = existing_metadata.get('total_proj_dim')
                logger.debug(f"Loaded layer dimensions from metadata")

            logger.info(f"Loaded metadata for {len(self.batch_info)} batches")

        except Exception as e:
            logger.error(f"Error loading metadata: {e}")

    def __del__(self):
        """Ensure metadata is saved on destruction."""
        try:
            if hasattr(self, '_pending_batches') and self._pending_batches:
                logger.info("Saving pending metadata on destruction")
                self.save_metadata()
        except Exception as e:
            logger.error(f"Error saving metadata during cleanup: {e}")