"""
Data module for efficient dataset operations.
"""

from .dataset import (
    TensorChunkedDataset,
    create_tensor_dataloader
)

__all__ = [
    "TensorChunkedDataset",
    "create_tensor_dataloader"
]