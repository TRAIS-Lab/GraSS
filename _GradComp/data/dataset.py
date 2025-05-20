"""
Dataset classes for loading gradients and influence function data.
"""

import os
import torch
import torch.utils.data
import re
import logging

# Configure logger
logger = logging.getLogger(__name__)

def custom_collate_fn(batch):
    """
    Custom collate function for gradient datasets.
    Instead of trying to stack tensors, it returns lists of batch indices and dictionaries.

    Args:
        batch: List of (batch_idx, batch_dict) tuples from the dataset

    Returns:
        Tuple of (batch_indices, batch_dicts) where:
        - batch_indices is a list of batch indices
        - batch_dicts is a list of dictionaries containing gradients or IFVP data
    """
    batch_indices = [item[0] for item in batch]
    batch_dicts = [item[1] for item in batch]
    return batch_indices, batch_dicts

class GradientDataset(torch.utils.data.Dataset):
    """Dataset for loading gradient or IFVP files from disk."""

    def __init__(self, disk_io, data_type="gradients", batch_range=None, is_test=False, layer_names=None):
        """
        Initialize dataset for loading gradient or IFVP files.

        Args:
            disk_io: DiskIOManager instance
            data_type: Type of data to load ("gradients" or "ifvp")
            batch_range: Optional tuple of (start_batch, end_batch) to filter files
            is_test: Whether to load test data files
            layer_names: List of layer names for optional validation
        """
        self.disk_io = disk_io
        self.data_type = data_type
        self.is_test = is_test
        self.layer_names = layer_names

        # Find all relevant files
        all_files = self.disk_io.find_batch_files(data_type, is_test)

        # Filter by batch range if needed
        if batch_range is not None:
            start_batch, end_batch = batch_range
            self.files = []
            for file_path in all_files:
                batch_idx = self.safe_extract_batch_idx(file_path)
                if batch_idx is not None and start_batch <= batch_idx < end_batch:
                    self.files.append(file_path)
        else:
            self.files = all_files

        # Sort files by batch index for consistency
        self.files = sorted(self.files, key=lambda f: self.safe_extract_batch_idx(f) or 0)
        logger.info(f"Found {len(self.files)} {data_type} files")

    def safe_extract_batch_idx(self, file_path):
        """
        Safely extract batch index from file path, handling both normal and test batch formats.

        Args:
            file_path: Path to the file

        Returns:
            Batch index as an integer, or None if it can't be extracted
        """
        try:
            # Try using the disk_io method first
            return self.disk_io.extract_batch_idx(file_path)
        except (ValueError, IndexError):
            # If that fails, try our custom extraction
            try:
                filename = os.path.basename(file_path)
                # Handle test_batch_X.mmap format
                if 'test_batch_' in filename:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        # Extract the number part after 'batch_'
                        return int(parts[2].split('.')[0])
                # Handle other potential formats
                elif 'batch' in filename:
                    # Find the number after 'batch' with regex
                    match = re.search(r'batch[_-]?(\d+)', filename)
                    if match:
                        return int(match.group(1))
                return None
            except (ValueError, IndexError):
                logger.warning(f"Could not extract batch index from {file_path}")
                return None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        batch_idx = self.safe_extract_batch_idx(file_path)
        batch_dict = self.disk_io.load_dict(file_path)
        return batch_idx, batch_dict