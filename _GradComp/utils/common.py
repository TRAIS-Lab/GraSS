"""
Common utility functions for gradient computation and influence attribution.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional

# Configure logger
logger = logging.getLogger(__name__)

def stable_inverse(matrix: torch.Tensor, damping: float = None) -> torch.Tensor:
    """
    Compute a numerically stable inverse of a matrix using eigendecomposition.

    Args:
        matrix: Input matrix to invert
        damping: (Adaptive) Damping factor for numerical stability

    Returns:
        Stable inverse of the input matrix with the same dtype as input
    """
    # Store original dtype for later conversion
    orig_dtype = matrix.dtype
    matrix = matrix.to(dtype=torch.float32)

    # Sometimes the matrix is a single number, so we need to check if it's a scalar
    if len(matrix.shape) == 0:
        if matrix == 0:
            # Return a 2d 0 tensor with same dtype
            return torch.tensor([[0.0]], device=matrix.device, dtype=orig_dtype)
        else:
            if damping is None:
                result = torch.tensor([[1.0 / (matrix * 1.1)]], device=matrix.device)
            else:
                result = torch.tensor([[1.0 / (matrix * (1 + damping))]], device=matrix.device)
            # Convert result back to original dtype
            return result.to(dtype=orig_dtype)

    # Add damping to the diagonal
    if damping is None:
        damping = 1e-2 * torch.trace(matrix) / matrix.size(0)
    else:
        damping = damping * torch.trace(matrix) / matrix.size(0)

    damped_matrix = matrix + damping * torch.eye(matrix.size(0), device=matrix.device)

    try:
        # Try Cholesky decomposition first (more stable)
        L = torch.linalg.cholesky(damped_matrix)
        inverse = torch.cholesky_inverse(L)
    except RuntimeError:
        logger.warning(f"Falling back to direct inverse due to Cholesky failure")
        # Fall back to direct inverse
        inverse = torch.inverse(damped_matrix)

    # Convert result back to the original dtype
    return inverse.to(dtype=orig_dtype)

def find_layers(model, layer_type="Linear", return_type="instance"):
    """
    Find layers of specified type in a model.

    Args:
        model: PyTorch model to search
        layer_type: Type of layer to find ('Linear', 'LayerNorm', or 'Linear_LayerNorm')
        return_type: What to return ('instance', 'name', or 'name_instance')

    Returns:
        List of layers, layer names, or (name, layer) tuples
    """
    layers = []
    return_module_name = not (return_type == "instance")

    if return_module_name:
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.LayerNorm) or isinstance(module, torch.nn.Embedding):
                layers.append((module_name, module))
    else:
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.LayerNorm) or isinstance(module, torch.nn.Embedding):
                layers.append(module)

    if return_module_name:
        if layer_type == "Linear":
            layers = [(name, layer) for name, layer in layers if isinstance(layer, torch.nn.Linear)]
        elif layer_type == "Linear_LayerNorm":
            layers = [(name, layer) for name, layer in layers if isinstance(layer, (torch.nn.Linear, torch.nn.LayerNorm))]
        elif layer_type == "LayerNorm":
            layers = [(name, layer) for name, layer in layers if isinstance(layer, torch.nn.LayerNorm)]
        else:
            raise ValueError("Invalid setting now. Choose from 'Linear', 'LayerNorm', and 'Linear_LayerNorm'.")
    else:
        if layer_type == "Linear":
            layers = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]
        elif layer_type == "Linear_LayerNorm":
            layers = [layer for layer in layers if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.LayerNorm)]
        elif layer_type == "LayerNorm":
            layers = [layer for layer in layers if isinstance(layer, torch.nn.LayerNorm)]
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

def vectorize(
    g: Dict[str, torch.Tensor],
    batch_dim: Optional[bool] = True,
    arr: Optional[torch.Tensor] = None,
    device: Optional[str] = "cuda",
) -> torch.Tensor:
    """
    Vectorize gradients into a flattened tensor.

    This function takes a dictionary of gradients and returns a flattened tensor
    of shape [batch_size, num_params].

    Args:
        g: A dictionary containing gradient tensors to be vectorized
        batch_dim: Whether to include the batch dimension in the returned tensor
        arr: An optional pre-allocated tensor to store the vectorized gradients
        device: The device to store the tensor on

    Returns:
        A flattened tensor of gradients
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

def get_parameter_chunk_sizes(
    param_shape_list: List,
    batch_size: int,
) -> tuple[int, List[int]]:
    """
    Compute chunk size information from feature to be projected.

    Get a tuple containing max chunk size and a list of the number of
    parameters in each chunk.

    Args:
        param_shape_list: A list of numbers indicating the total number of
            features to be projected. A typical example is a list of parameter
            size of each module in a torch.nn.Module model.
        batch_size: The batch size. Each term (or module) in feature
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
    params_per_chunk = [max_chunk_size] * num_chunk + [remaining] if remaining > 0 else [max_chunk_size] * num_chunk

    return max_chunk_size, params_per_chunk

def aggregate_influence_scores(results_dir, output_file=None, total_train_samples=None, num_test=None):
    """
    Aggregate partial influence scores saved by the attribute method.

    Args:
        results_dir: Directory containing partial result files
        output_file: Path to save the aggregated results
        total_train_samples: Total number of training samples
        num_test: Number of test samples

    Returns:
        Aggregated influence scores tensor
    """
    from tqdm import tqdm

    logger.info(f"Aggregating influence scores from {results_dir}...")

    # Find all partial result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.pt')]

    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")

    # Load the first file to get metadata if not provided
    first_file = os.path.join(results_dir, result_files[0])
    first_data = torch.load(first_file)

    if total_train_samples is None or num_test is None:
        if 'metadata' in first_data:
            metadata = first_data['metadata']
            total_train_samples = metadata.get('total_train_samples')
            num_test = metadata.get('num_test')

        if total_train_samples is None or num_test is None:
            raise ValueError("Could not determine total_train_samples or num_test. Please provide these values.")

    # Initialize the full results tensor
    IF_score = torch.zeros(total_train_samples, num_test)

    # Aggregate all partial results
    for file_name in tqdm(result_files, desc="Aggregating files"):
        file_path = os.path.join(results_dir, file_name)
        data = torch.load(file_path)

        if 'metadata' in data and 'scores' in data:
            scores = data['scores']
            metadata = data['metadata']
            min_sample = metadata['min_sample']
            max_sample = metadata['max_sample']

            # Add this chunk's scores to the full tensor
            IF_score[min_sample:max_sample, :] += scores
        else:
            # Fall back to parsing file name for older format
            parts = file_name.split('_')
            if len(parts) >= 4:
                min_sample = int(parts[-2])
                max_sample = int(parts[-1].split('.')[0])
                scores = data
                IF_score[min_sample:max_sample, :] += scores

    # Save aggregated results if output file is specified
    if output_file is not None:
        torch.save(IF_score, output_file)
        logger.info(f"Saved aggregated results to {output_file}")

    return IF_score