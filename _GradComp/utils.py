"""Utility functions for working with models and parameters. Adapted from dattri"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple

import numpy as np

import time
import json
import torch
import torch.nn as nn
from torch import Tensor

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

def _vectorize(
    g: Dict[str, Tensor],
    batch_dim: Optional[bool] = True,
    arr: Optional[Tensor] = None,
    device: Optional[str] = "cuda",
) -> Tensor:
    """Vectorize gradients into a flattened tensor.

    This function takes a dictionary of gradients and returns a flattened tensor
    of shape [batch_size, num_params].

    Args:
        g (Dict[str, Tensor]): A dictionary containing gradient tensors to be
            vectorized.
        batch_dim (bool, optional): Whether to include the batch dimension in the
            returned tensor. Defaults to True.
        arr (Tensor, optional): An optional pre-allocated tensor to store the
            vectorized gradients. If provided, it must have the shape
            `[batch_size, num_params]`, where `num_params` is the total number of
            scalar parameters in all the tensors in `g`. If not provided, a new
            tensor will be allocated. Defaults to None.
        device (str, optional): The device to store the tensor on. Either "cuda"
            or "cpu". Defaults to "cuda".

    Returns:
        Tensor: A flattened tensor of gradients. If batch_dim is True, shape is
        `[batch_size, num_params]`, where each row contains all the vectorized
        gradients for a single element in the batch. Otherwise, shape is
        `[num_params]`.

    Raises:
        ValueError: If parameter size in g doesn't match batch size.
    """
    if arr is None:
        if batch_dim:
            g_elt = g[next(iter(g.keys()))]
            batch_size = g_elt.shape[0]
            num_params = 0
            for param in g.values():
                if param.shape[0] != batch_size:
                    msg = "Parameter row num doesn't match batch size."
                    raise ValueError(msg)
                num_params += int(param.numel() / batch_size)
            arr = torch.empty(
                size=(batch_size, num_params),
                dtype=g_elt.dtype,
                device=device,
            )
        else:
            num_params = 0
            for param in g.values():
                num_params += int(param.numel())
            arr = torch.empty(size=(num_params,), dtype=param.dtype, device=device)

    pointer = 0
    vector_dim = 1
    for param in g.values():
        if batch_dim:
            if len(param.shape) <= vector_dim:
                num_param = 1
                p = param.data.reshape(-1, 1)
            else:
                num_param = param[0].numel()
                p = param.flatten(start_dim=1).data
            arr[:, pointer : pointer + num_param] = p.to(device)
            pointer += num_param
        else:
            num_param = param.numel()
            arr[pointer : pointer + num_param] = param.reshape(-1).to(device)
            pointer += num_param
    return arr


def _get_parameter_chunk_sizes(
    param_shape_list: List,
    batch_size: int,
) -> tuple[int, int]:
    """Compute chunk size information from feature to be projected.

    Get a tuple containing max chunk size and a list of the number of
    parameters in each chunk.

    Args:
        param_shape_list (List): A list of numbers indicating the total number of
            features to be projected. A typical example is a list of parameter
            size of each module in a torch.nn.Module model.
        batch_size (int): The batch size. Each term (or module) in feature
            will have the same batch size.

    Returns:
        tuple[int, List[int]]: A tuple containing:
            - Maximum number of parameters per chunk
            - A list of the number of parameters in each chunk
    """
    # get the number of params of each term in feature
    param_shapes = np.array(param_shape_list)

    chunk_sum = 0
    max_chunk_size = np.iinfo(np.uint32).max // batch_size
    params_per_chunk = []

    for ps in param_shapes:
        if chunk_sum + ps >= max_chunk_size:
            params_per_chunk.append(chunk_sum)
            chunk_sum = 0

        chunk_sum += ps

    if param_shapes.sum() - np.sum(params_per_chunk) > 0:
        params_per_chunk.append(param_shapes.sum() - np.sum(params_per_chunk))

    return max_chunk_size, params_per_chunk


def get_parameter_chunk_sizes(
    param_shape_list: List,
    batch_size: int,
) -> tuple[int, int]:
    """Compute chunk size information from feature to be projected.

    Get a tuple containing max chunk size and a list of the number of
    parameters in each chunk.

    Args:
        param_shape_list (List): A list of numbers indicating the total number of
            features to be projected. A typical example is a list of parameter
            size of each module in a torch.nn.Module model.
        batch_size (int): The batch size. Each term (or module) in feature
            will have the same batch size.

    Returns:
        tuple[int, List[int]]: A tuple containing:
            - Maximum number of parameters per chunk
            - A list of the number of parameters in each chunk
    """
    # get the number of total params
    param_num = param_shape_list[0]

    max_chunk_size = np.iinfo(np.uint32).max // batch_size

    num_chunk = param_num // max_chunk_size
    remaining = param_num % max_chunk_size
    params_per_chunk = [max_chunk_size] * num_chunk + [remaining]

    return max_chunk_size, params_per_chunk

def stable_inverse(matrix: torch.Tensor, damping: float = None) -> torch.Tensor:
    """
    Compute a numerically stable inverse of a matrix using eigendecomposition.

    Args:
        matrix: Input matrix to invert
        damping: Damping factor for numerical stability

    Returns:
        Stable inverse of the input matrix
    """
    # sometimes the matrix is a single number, so we need to check if it's a scalar
    if len(matrix.shape) == 0:
        if matrix == 0:
            # return a 2d 0 tensor
            return torch.tensor([[0.0]], device=matrix.device)
        else:
            if damping is None:
                return torch.tensor([[1.0 / (matrix * 1.1)]], device=matrix.device)
            else:
                return torch.tensor([[1.0 / (matrix * (1 + damping))]], device=matrix.device)

    # Add damping to the diagonal
    if damping is None:
        damping = 0.1 * torch.trace(matrix) / matrix.size(0)

    damped_matrix = matrix + damping * torch.eye(matrix.size(0), device=matrix.device)

    try:
        # Try Cholesky decomposition first (more stable)
        L = torch.linalg.cholesky(damped_matrix)
        inverse = torch.cholesky_inverse(L)
    except RuntimeError:
        print(f"Falling back to direct inverse due to Cholesky failure")
        # Fall back to direct inverse
        inverse = torch.inverse(damped_matrix)

    return inverse

def find_layers(model, layer_type="Linear", return_type="instance"):
    layers = []
    return_module_name = not (return_type == "instance")

    if return_module_name:
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm) or isinstance(module, nn.Embedding):
                layers.append((module_name, module))
    else:
        for module in model.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm) or isinstance(module, nn.Embedding):
                layers.append(module)

    if return_module_name:
        if layer_type == "Linear":
            layers = [(name, layer) for name, layer in layers if isinstance(layer, nn.Linear)]
        elif layer_type == "Linear_LayerNorm":
            layers = [(name, layer) for name, layer in layers if isinstance(layer, (nn.Linear, nn.LayerNorm))]
        elif layer_type == "LayerNorm":
            layers = [(name, layer) for name, layer in layers if isinstance(layer, nn.LayerNorm)]
        else:
            raise ValueError("Invalid setting now. Choose from 'Linear', 'LayerNorm', and 'Linear_LayerNorm'.")
    else:
        if layer_type == "Linear":
            layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
        elif layer_type == "Linear_LayerNorm":
            layers = [layer for layer in layers if isinstance(layer, nn.Linear) or isinstance(layer, nn.LayerNorm)]
        elif layer_type == "LayerNorm":
            layers = [layer for layer in layers if isinstance(layer, nn.LayerNorm)]
        else:
            raise ValueError("Invalid setting now. Choose from 'Linear', 'LayerNorm', and 'Linear_LayerNorm'.")

    if return_type == "instance":
        return layers
    elif return_type == "name":
        return [name for name, layer in layers]
    elif return_type == "name_instance":
        return [(name, layer) for name, layer in layers]
    else:
        raise ValueError("Invalid return_type. Choose from 'instance', 'name', and 'name_instance'.")