"""
Metadata management for batches, layers, and processing state.
"""

import os
import json
import time
import logging
from typing import Dict, List, Tuple, Optional

# Configure logger
logger = logging.getLogger(__name__)

class MetadataManager:
    """
    Manager for batch metadata (sample counts, indices, etc.)
    This class avoids nested dictionaries in favor of flat structures.
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

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            # Check for existing metadata
            self._load_metadata_if_exists()

        logger.info(f"Initialized MetadataManager with {len(layer_names)} layers")

    def add_batch_info(self, batch_idx: int, sample_count: int) -> None:
        """
        Add information about a batch.

        Args:
            batch_idx: Index of the batch
            sample_count: Number of samples in the batch
        """
        self.batch_info[batch_idx] = {
            "sample_count": sample_count,
            "start_idx": self.total_samples
        }
        self.total_samples += sample_count
        logger.debug(f"Added batch {batch_idx} with {sample_count} samples (total: {self.total_samples})")

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
        return {
            batch_idx: (info["start_idx"], info["start_idx"] + info["sample_count"])
            for batch_idx, info in self.batch_info.items()
        }

    def save_metadata(self) -> None:
        """
        Save metadata to disk with file locking, supporting out-of-order batch processing.
        """
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
                        logger.error(f"Error loading existing metadata: {e}")

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

                logger.info(f"Saved metadata for {len(self.batch_info)} batches to {metadata_path}")
        except FileExistsError:
            # Lock already exists, wait and retry
            logger.warning("Metadata is being updated by another process, waiting...")
            time.sleep(1)
            self.save_metadata()  # Recursive retry
        finally:
            # Remove lock file if it exists and we created it
            if os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
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
                    if 'layer_names' in metadata:
                        self.layer_names = metadata['layer_names']

                logger.info(f"Loaded metadata for {len(self.batch_info)} batches from {metadata_path}")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")

    def get_batch_indices(self) -> List[int]:
        """Get all batch indices."""
        return sorted(self.batch_info.keys())