"""
Enhanced metadata management with worker coordination support.
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
    Enhanced metadata manager with worker coordination support.
    Supports pre-computing full dataset info and coordinated partial updates.
    """

    def __init__(self, cache_dir: str, layer_names: List[str]):
        self.cache_dir = cache_dir
        self.layer_names = layer_names
        self.batch_info = {}  # Maps batch_idx -> {sample_count, start_idx}
        self.total_samples = 0
        self.layer_dims = None
        self.total_proj_dim = None

        # Full dataset information (computed once, shared by all workers)
        self.dataset_info: Optional[DatasetInfo] = None

        self._metadata_lock = threading.Lock()
        self._pending_batches = {}
        self._last_save_time = 0
        self._save_interval = 5.0

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_metadata_if_exists()

        logger.debug(f"Initialized EnhancedMetadataManager with {len(layer_names)} layers")

    def initialize_full_dataset(self, train_dataloader) -> None:
        """
        Initialize full dataset metadata from dataloader.
        Should be called once before parallel processing begins.
        """
        if self.dataset_info is not None:
            logger.info("Dataset info already initialized")
            return

        total_batches = len(train_dataloader)
        batch_size = train_dataloader.batch_size

        # Compute total samples by iterating through dataset once
        # This is more accurate than batch_size * total_batches due to drop_last behavior
        logger.info("Computing full dataset information...")

        batch_to_sample_mapping = {}
        current_sample_idx = 0

        # We need to be careful here - we don't want to actually run the model
        # Just get the batch sizes
        if hasattr(train_dataloader.dataset, '__len__'):
            # For most datasets, we can compute this without iteration
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
            # Fallback: assume all batches have batch_size except possibly the last
            logger.warning("Dataset doesn't support __len__, using fallback batch size computation")
            for batch_idx in range(total_batches):
                batch_to_sample_mapping[batch_idx] = (current_sample_idx, current_sample_idx + batch_size)
                current_sample_idx += batch_size

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

        # Save the full dataset metadata immediately
        self._save_full_dataset_metadata()

        logger.info(f"Initialized full dataset: {total_batches} batches, {current_sample_idx} samples")

    def _save_full_dataset_metadata(self) -> None:
        """Save the complete dataset metadata with file locking."""
        if not self.cache_dir or self.dataset_info is None:
            return

        metadata_path = self._get_metadata_path()
        temp_path = metadata_path + ".tmp"

        try:
            # Prepare complete metadata
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
                'timestamp': time.time(),
                'metadata_version': '2.0'  # Version to distinguish from old format
            }

            # Atomic write with file locking
            with open(temp_path, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(metadata, f, separators=(',', ':'))
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_path, metadata_path)
            logger.info("Saved full dataset metadata")

        except Exception as e:
            logger.error(f"Error saving full dataset metadata: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def update_worker_batch_info(self, batch_idx: int, sample_count: int, worker_id: str) -> None:
        """
        Update batch info from a specific worker (for validation/detailed tracking).
        This doesn't change the overall structure but can add worker-specific details.
        """
        with self._metadata_lock:
            if batch_idx in self.batch_info:
                # Validate that the sample count matches expectations
                expected_count = self.batch_info[batch_idx]["sample_count"]
                if sample_count != expected_count:
                    logger.warning(f"Worker {worker_id}: Batch {batch_idx} sample count mismatch. "
                                 f"Expected: {expected_count}, Got: {sample_count}")

                # Add worker-specific info
                self.batch_info[batch_idx][f"worker_{worker_id}"] = {
                    "processed": True,
                    "sample_count": sample_count,
                    "timestamp": time.time()
                }
            else:
                logger.error(f"Worker {worker_id}: Trying to update unknown batch {batch_idx}")

    def add_batch_info(self, batch_idx: int, sample_count: int) -> None:
        """Backward compatibility method."""
        # Extract worker ID from current context if available
        import threading
        thread_name = threading.current_thread().name
        worker_id = thread_name if "worker" in thread_name.lower() else "unknown"
        self.update_worker_batch_info(batch_idx, sample_count, worker_id)

    def get_total_samples(self) -> int:
        """Get the total number of samples across all batches."""
        if self.dataset_info:
            return self.dataset_info.total_samples
        return self.total_samples

    def get_batch_to_sample_mapping(self) -> Dict[int, Tuple[int, int]]:
        """Get mapping from batch indices to sample index ranges."""
        if self.dataset_info:
            return self.dataset_info.batch_to_sample_mapping

        # Fallback to old method
        return {
            batch_idx: (info["start_idx"], info["start_idx"] + info["sample_count"])
            for batch_idx, info in self.batch_info.items()
        }

    def get_total_batches(self) -> int:
        """Get total number of batches in the dataset."""
        if self.dataset_info:
            return self.dataset_info.total_batches
        return len(self.batch_info)

    def set_layer_dims(self, layer_dims: List[int]) -> None:
        """Set the layer dimensions."""
        with self._metadata_lock:
            self.layer_dims = layer_dims
            self.total_proj_dim = sum(layer_dims) if layer_dims else None
            logger.debug(f"Set layer dimensions: {len(layer_dims)} layers, total={self.total_proj_dim}")

            # Update the saved metadata if we have full dataset info
            if self.dataset_info is not None:
                self._save_full_dataset_metadata()

    def save_metadata(self) -> None:
        """
        Save metadata - now only updates worker-specific info if full dataset exists.
        """
        if not self.cache_dir:
            return

        if self.dataset_info is not None:
            # We have full dataset info, just update it
            self._save_full_dataset_metadata()
        else:
            # Fallback to old behavior for backward compatibility
            self._save_legacy_metadata()

    def _save_legacy_metadata(self) -> None:
        """Legacy metadata saving for backward compatibility."""
        metadata_path = self._get_metadata_path()
        temp_path = metadata_path + ".tmp"

        try:
            with self._metadata_lock:
                self._flush_pending_batches()

            metadata = {
                'batch_info': {str(k): v for k, v in self.batch_info.items()},
                'layer_names': self.layer_names,
                'layer_dims': self.layer_dims,
                'total_proj_dim': self.total_proj_dim,
                'total_samples': self.total_samples,
                'timestamp': time.time(),
                'metadata_version': '1.0'
            }

            with open(temp_path, 'w') as f:
                json.dump(metadata, f, separators=(',', ':'))

            os.replace(temp_path, metadata_path)
            logger.debug(f"Saved legacy metadata for {len(self.batch_info)} batches")

        except Exception as e:
            logger.error(f"Error saving legacy metadata: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

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
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Handle both new and legacy formats
                version = metadata.get('metadata_version', '1.0')

                if version == '2.0' and 'dataset_info' in metadata:
                    # New format with full dataset info
                    self._load_v2_metadata(metadata)
                else:
                    # Legacy format
                    self._load_legacy_metadata(metadata)

                logger.info(f"Loaded metadata (version {version}) for {len(self.batch_info)} batches")

            except Exception as e:
                logger.error(f"Error loading metadata: {e}")

    def _load_v2_metadata(self, metadata: dict) -> None:
        """Load version 2.0 metadata with full dataset info."""
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

        # Load full dataset info
        dataset_info_dict = metadata['dataset_info']
        self.dataset_info = DatasetInfo(
            total_batches=dataset_info_dict['total_batches'],
            total_samples=dataset_info_dict['total_samples'],
            batch_size=dataset_info_dict['batch_size'],
            batch_to_sample_mapping={
                int(k): v for k, v in dataset_info_dict['batch_to_sample_mapping'].items()
            }
        )

        logger.debug("Loaded full dataset information from metadata")

    def _load_legacy_metadata(self, metadata: dict) -> None:
        """Load legacy metadata format."""
        self.batch_info = {
            int(batch_idx): info for batch_idx, info in metadata['batch_info'].items()
        }

        self.total_samples = metadata.get('total_samples', 0)

        if 'layer_names' in metadata:
            self.layer_names = metadata['layer_names']
        if 'layer_dims' in metadata:
            self.layer_dims = metadata['layer_dims']
            self.total_proj_dim = metadata.get('total_proj_dim')

    def __del__(self):
        """Ensure metadata is saved on destruction."""
        try:
            if hasattr(self, '_pending_batches') and self._pending_batches:
                logger.info("Saving pending metadata on destruction")
                self.save_metadata()
        except Exception as e:
            logger.error(f"Error saving metadata during cleanup: {e}")