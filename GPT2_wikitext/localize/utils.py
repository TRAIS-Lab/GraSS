import torch
from transformers import AutoModelForCausalLM, logging as transformers_logging
import re
import logging
from functools import lru_cache

# Cache for model loading to prevent repeated loads
MODEL_CACHE = {}

def active_localize_indices(mask_path, module_name, model_name_or_path=None, device="cuda"):
    """
    Extract active indices for a specific module from localization mask,
    accounting for Conv1D/Linear transposition in the original model architecture.

    Args:
        mask_path: Path to the mask file (.pt)
        module_name: Name of the module (e.g., "transformer.ln_f")
        model_name_or_path: Path or name of the original model to check for Conv1D layers
        device: Device to load the mask on

    Returns:
        active_indices: Tensor of indices that are active (unmasked)
    """
    # Load mask dictionary
    mask_dict = torch.load(mask_path, map_location=device)
    binary_masks = mask_dict["binary_mask"]
    param_names = mask_dict["trainable_names"]

    # Initialize masks as None
    weight_mask = None
    bias_mask = None

    # Find the exact parameter names for this module
    for i, name in enumerate(param_names):
        # Check for exact match with weight/bias suffix
        if name == f"{module_name}.weight":
            weight_mask = binary_masks[i]
        elif name == f"{module_name}.bias":
            bias_mask = binary_masks[i]

        # Special handling for attention modules that have combined weights
        if module_name.endswith("attn.c_attn") and name.endswith("attn.c_attn.weight"):
            weight_mask = binary_masks[i]
        elif module_name.endswith("attn.c_attn") and name.endswith("attn.c_attn.bias"):
            bias_mask = binary_masks[i]

    # If no masks found, return None
    if weight_mask is None and bias_mask is None:
        logging.warning(f"No mask found for {module_name}")
        return None

    # Check if the module uses Conv1D in the original model
    is_conv1d = False
    if model_name_or_path is not None:
        is_conv1d = check_if_conv1d(model_name_or_path, module_name, device)
        if is_conv1d:
            logging.info(f"Detected Conv1D layer for {module_name}, adjusting mask orientation")
            if weight_mask is not None:
                # Transpose the mask to match the actual parameter orientation
                weight_mask = weight_mask.transpose(0, 1)

    # Different handling based on module type
    if "ln_" in module_name or module_name.endswith("ln_f"):
        # For LayerNorm, we want indices where either weight or bias is active
        if weight_mask is not None and bias_mask is not None:
            # Convert to boolean before using bitwise operations
            weight_bool = weight_mask > 0.5
            bias_bool = bias_mask > 0.5
            active_indices = torch.where(weight_bool | bias_bool)[0]
        elif weight_mask is not None:
            active_indices = torch.where(weight_mask > 0.5)[0]
        else:
            active_indices = torch.where(bias_mask > 0.5)[0]

    elif "attn.c_attn" in module_name:
        # For query/key/value projection, we want to handle the 3-way split
        if weight_mask is not None:
            if is_conv1d:
                # In Conv1D: weight is [3*hidden_size, hidden_size]
                # Active rows represent output features
                row_active = (weight_mask.sum(dim=1) > 0)
                active_indices = torch.where(row_active)[0]
            else:
                # In Linear: weight is [hidden_size, 3*hidden_size]
                # Active columns represent output features
                col_active = (weight_mask.sum(dim=0) > 0)
                active_indices = torch.where(col_active)[0]

    elif "attn.c_proj" in module_name:
        # For attention output projection
        if weight_mask is not None:
            if is_conv1d:
                # In Conv1D: weight is [hidden_size, hidden_size]
                # Active columns represent input features
                col_active = (weight_mask.sum(dim=0) > 0)
                active_indices = torch.where(col_active)[0]
            else:
                # In Linear: weight is [hidden_size, hidden_size]
                # Active columns represent input features
                col_active = (weight_mask.sum(dim=0) > 0)
                active_indices = torch.where(col_active)[0]

    elif "mlp.c_fc" in module_name:
        # For MLP expansion layer
        if weight_mask is not None:
            if is_conv1d:
                # In Conv1D: weight is [4*hidden_size, hidden_size]
                # Active rows represent output features
                row_active = (weight_mask.sum(dim=1) > 0)
                active_indices = torch.where(row_active)[0]
            else:
                # In Linear: weight is [hidden_size, 4*hidden_size]
                # Active columns represent output features
                col_active = (weight_mask.sum(dim=0) > 0)
                active_indices = torch.where(col_active)[0]

    elif "mlp.c_proj" in module_name:
        # For MLP projection layer
        if weight_mask is not None:
            if is_conv1d:
                # In Conv1D: weight is [hidden_size, 4*hidden_size]
                # Active columns represent input features
                col_active = (weight_mask.sum(dim=0) > 0)
                active_indices = torch.where(col_active)[0]
            else:
                # In Linear: weight is [4*hidden_size, hidden_size]
                # Active columns represent input features (rows in original)
                col_active = (weight_mask.sum(dim=0) > 0)
                active_indices = torch.where(col_active)[0]

    elif module_name == "transformer.wte" or module_name == "transformer.wpe":
        # For embeddings, we want active embedding dimensions (columns)
        if weight_mask is not None:
            # Get active column indices
            col_active = (weight_mask.sum(dim=0) > 0)
            active_indices = torch.where(col_active)[0]

    elif module_name == "lm_head":
        # For language model head, we want active input features
        if weight_mask is not None:
            # Get active column indices
            col_active = (weight_mask.sum(dim=0) > 0)
            active_indices = torch.where(col_active)[0]

    else:
        # Default case
        if weight_mask is not None:
            if is_conv1d:
                # For Conv1D, rows are output features
                row_active = (weight_mask.sum(dim=1) > 0)
                active_indices = torch.where(row_active)[0]
            else:
                # For Linear, rows are usually output features
                row_active = (weight_mask.sum(dim=1) > 0)
                active_indices = torch.where(row_active)[0]
        else:
            # Only bias exists
            active_indices = torch.where(bias_mask > 0.5)[0]

    # Verify we have some active indices
    if active_indices is None or len(active_indices) == 0:
        logging.warning(f"No active indices found for {module_name}")

    return active_indices


@lru_cache(maxsize=2)  # Cache up to 2 model configurations
def load_model_with_suppressed_logs(model_name_or_path, device="cuda"):
    """
    Load a model with suppressed logging, and cache the result.

    Args:
        model_name_or_path: Path or name of the model
        device: Device to load the model on

    Returns:
        The loaded model
    """
    # Check if model is already loaded
    if model_name_or_path in MODEL_CACHE:
        return MODEL_CACHE[model_name_or_path]

    # Suppress the HF logging
    original_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()  # Only show errors

    try:
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True)
        model.to(device)
        # Cache the model
        MODEL_CACHE[model_name_or_path] = model
        return model
    finally:
        # Restore the original verbosity
        transformers_logging.set_verbosity(original_verbosity)


def check_if_conv1d(model_name_or_path, module_name, device="cuda"):
    """
    Check if a module in the original model uses Conv1D implementation.

    Args:
        model_name_or_path: Path or name of the original model
        module_name: Name of the module to check
        device: Device to load the model on

    Returns:
        bool: True if the module uses Conv1D, False otherwise
    """
    try:
        # Load the model with suppressed logs
        model = load_model_with_suppressed_logs(model_name_or_path, device)

        # Use regex to extract the layer index for transformer blocks
        layer_match = re.search(r'transformer\.h\.(\d+)', module_name)

        if layer_match:
            # For transformer blocks, navigate down the hierarchy
            layer_idx = int(layer_match.group(1))

            # Handle attention layers
            if "attn.c_attn" in module_name:
                module = model.transformer.h[layer_idx].attn.c_attn
                return hasattr(module, 'conv1d') or 'Conv1D' in str(type(module))

            elif "attn.c_proj" in module_name:
                module = model.transformer.h[layer_idx].attn.c_proj
                return hasattr(module, 'conv1d') or 'Conv1D' in str(type(module))

            elif "mlp.c_fc" in module_name:
                module = model.transformer.h[layer_idx].mlp.c_fc
                return hasattr(module, 'conv1d') or 'Conv1D' in str(type(module))

            elif "mlp.c_proj" in module_name:
                module = model.transformer.h[layer_idx].mlp.c_proj
                return hasattr(module, 'conv1d') or 'Conv1D' in str(type(module))

        # Special cases for other modules
        if module_name == "lm_head":
            if hasattr(model, 'lm_head'):
                return hasattr(model.lm_head, 'conv1d') or 'Conv1D' in str(type(model.lm_head))
            else:
                return False

    except Exception as e:
        logging.error(f"Error checking Conv1D for {module_name}: {e}")
        return False

    # Default assumption: it's not Conv1D
    return False