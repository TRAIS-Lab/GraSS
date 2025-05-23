"""
Metadata management for batches, layers, and processing state.
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
    Manager for batch metadata with support for layer dimensions.
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

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            # Check for existing metadata
            self._load_metadata_if_exists()

        logger.debug(f"Initialized MetadataManager with {len(layer_names)} layers")

    def add_batch_info(self, batch_idx: int, sample_count: int) -> None:
        """
        Add information about a batch with optimized batching.

        Args:
            batch_idx: Index of the batch
            sample_count: Number of samples in the batch
        """
        with self._metadata_lock:
            # Add to pending buffer
            self._pending_batches[batch_idx] = {
                "sample_count": sample_count,
                "start_idx": self.total_samples
            }
            self.total_samples += sample_count

            # Check if we should save (time-based or buffer size-based)
            current_time = time.time()
            should_save = (
                len(self._pending_batches) >= 100 or  # Buffer size threshold
                (current_time - self._last_save_time) >= self._save_interval  # Time threshold
            )

            if should_save:
                self._flush_pending_batches()

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

        # Merge pending batches
        self.batch_info.update(self._pending_batches)
        self._pending_batches.clear()
        self._last_save_time = time.time()

    def get_total_samples(self) -> int:
        """
        Get the total number of samples across all batches.

        Returns:
            Total number of samples
        """
        return self.total_samples

    def get_batch_to_sample_mapping(self) -> Dict[int, Tuple[int, int]]:
        """
        Get mapping from batch indices to sample index ranges.

        Returns:
            Dictionary mapping batch indices to (start_idx, end_idx) tuples
        """
        # Ensure pending batches are flushed
        with self._metadata_lock:
            self._flush_pending_batches()

        return {
            batch_idx: (info["start_idx"], info["start_idx"] + info["sample_count"])
            for batch_idx, info in self.batch_info.items()
        }

    def save_metadata(self) -> None:
        """
        Save metadata to disk with optimized approach and reduced lock contention.
        """
        if not self.cache_dir:
            return

        with self._metadata_lock:
            # Flush any pending batches first
            self._flush_pending_batches()

        metadata_path = self._get_metadata_path()
        temp_path = metadata_path + ".tmp"

        try:
            # Create the metadata structure
            # Recompute start indices to ensure consistency
            sorted_batches = sorted(self.batch_info.keys())
            current_idx = 0
            batch_info_corrected = {}

            for batch_idx in sorted_batches:
                sample_count = self.batch_info[batch_idx]["sample_count"]
                batch_info_corrected[batch_idx] = {
                    "start_idx": current_idx,
                    "sample_count": sample_count
                }
                current_idx += sample_count

            # Prepare serializable format (convert int keys to strings for JSON)
            serializable_info = {
                str(idx): info for idx, info in batch_info_corrected.items()
            }

            metadata = {
                'batch_info': serializable_info,
                'layer_names': self.layer_names,
                'layer_dims': self.layer_dims,  # Save layer dimensions
                'total_proj_dim': self.total_proj_dim,  # Save total projection dimension
                'total_samples': current_idx,
                'version': '2.0',  # Updated version for tensor format
                'timestamp': time.time()
            }

            # Write to temporary file first (atomic operation)
            with open(temp_path, 'w') as f:
                json.dump(metadata, f, separators=(',', ':'))  # Compact format

            # Atomic rename
            os.replace(temp_path, metadata_path)

            # Update our internal state to match what we saved
            with self._metadata_lock:
                self.batch_info = batch_info_corrected
                self.total_samples = current_idx

            logger.debug(f"Saved metadata for {len(self.batch_info)} batches with layer_dims: {self.layer_dims}")

        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

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

                # Load layer information
                if 'layer_names' in metadata:
                    self.layer_names = metadata['layer_names']

                # Load layer dimensions (new in version 2.0)
                if 'layer_dims' in metadata:
                    self.layer_dims = metadata['layer_dims']
                    self.total_proj_dim = metadata.get('total_proj_dim')
                    logger.debug(f"Loaded layer dimensions from metadata")

                logger.info(f"Loaded metadata for {len(self.batch_info)} batches")

            except Exception as e:
                logger.error(f"Error loading metadata: {e}")

    def get_batch_indices(self) -> List[int]:
        """Get all batch indices."""
        with self._metadata_lock:
            # Include both saved and pending batches
            all_indices = set(self.batch_info.keys())
            all_indices.update(self._pending_batches.keys())
            return sorted(all_indices)

    def force_save(self) -> None:
        """Force immediate save of all pending data."""
        self.save_metadata()

    def __del__(self):
        """Ensure metadata is saved on destruction."""
        try:
            if hasattr(self, '_pending_batches') and self._pending_batches:
                logger.info("Saving pending metadata on destruction")
                self.save_metadata()
        except Exception as e:
            logger.error(f"Error saving metadata during cleanup: {e}")