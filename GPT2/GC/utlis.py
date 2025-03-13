import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '...'))
sys.path.append(parent_dir)

from _gradcomp.layers.linear import GCLinear, GCEmbedding
from _gradcomp.layers.layer_norm import GCLayerNorm

# For GPT2 specifically
Conv1DSwitch = [
    '.attn.c_attn.weight',
    '.attn.c_proj.weight',
    '.mlp.c_fc.weight',
    '.mlp.c_proj.weight',
]

def transpose_Conv1D(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if any(pattern in key for pattern in Conv1DSwitch):
            new_state_dict[key] = value.t()
        else:
            new_state_dict[key] = value
    return new_state_dict

def find_GClayers(model, setting="Linear"):
    GC_layers = []

    for module in model.modules():
        if isinstance(module, GCLinear) or isinstance(module, GCLayerNorm) or isinstance(module, GCEmbedding):
            GC_layers.append(module)

    if setting == "Linear":
        GC_layers = [layer for layer in GC_layers if isinstance(layer, GCLinear)]
    elif setting == "Linear_LayerNorm":
        GC_layers = [layer for layer in GC_layers if isinstance(layer, GCLinear) or isinstance(layer, GCLayerNorm)]
    elif setting == "LayerNorm":
        GC_layers = [layer for layer in GC_layers if isinstance(layer, GCLayerNorm)]
    else:
        raise ValueError("Invalid setting now. Choose from 'Linear', 'LayerNorm', and 'Linear_LayerNorm'.")

    return GC_layers