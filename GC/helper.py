from .layers.linear import GCLinear, GCEmbedding
from .layers.layer_norm import GCLayerNorm

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

def find_GClayers(model):
    GC_layers = []

    for module in model.modules():
        if isinstance(module, GCLinear) or isinstance(module, GCLayerNorm) or isinstance(module, GCEmbedding):
            GC_layers.append(module)

    return GC_layers