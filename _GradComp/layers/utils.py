from .linear import GCLinear, GCEmbedding
from .layer_norm import GCLayerNorm

def find_GClayers(model, setting="Linear", return_module_name=False):
    GC_layers = []

    if return_module_name:
        for module_name, module in model.named_modules():
            if isinstance(module, GCLinear) or isinstance(module, GCLayerNorm) or isinstance(module, GCEmbedding):
                GC_layers.append((module_name, module))
    else:
        for module in model.modules():
            if isinstance(module, GCLinear) or isinstance(module, GCLayerNorm) or isinstance(module, GCEmbedding):
                GC_layers.append(module)

    if return_module_name:
        if setting == "Linear":
            GC_layers = [(name, layer) for name, layer in GC_layers if isinstance(layer, GCLinear)]
        elif setting == "Linear_LayerNorm":
            GC_layers = [(name, layer) for name, layer in GC_layers if isinstance(layer, (GCLinear, GCLayerNorm))]
        elif setting == "LayerNorm":
            GC_layers = [(name, layer) for name, layer in GC_layers if isinstance(layer, GCLayerNorm)]
        else:
            raise ValueError("Invalid setting now. Choose from 'Linear', 'LayerNorm', and 'Linear_LayerNorm'.")
    else:
        if setting == "Linear":
            GC_layers = [layer for layer in GC_layers if isinstance(layer, GCLinear)]
        elif setting == "Linear_LayerNorm":
            GC_layers = [layer for layer in GC_layers if isinstance(layer, GCLinear) or isinstance(layer, GCLayerNorm)]
        elif setting == "LayerNorm":
            GC_layers = [layer for layer in GC_layers if isinstance(layer, GCLayerNorm)]
        else:
            raise ValueError("Invalid setting now. Choose from 'Linear', 'LayerNorm', and 'Linear_LayerNorm'.")

    return GC_layers