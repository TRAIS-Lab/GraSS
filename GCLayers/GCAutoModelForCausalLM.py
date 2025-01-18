import os
import json
from typing import Union


import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM
from transformers.models.gpt2.modeling_gpt2 import Conv1D

from .linear import GCLinear

class GCAutoModelForCausalLM(AutoModelForCausalLM):
    """Custom AutoModelForCausalLM that automatically replaces standard layers with GC layers."""
    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        save_function = torch.save,
        **kwargs,
    ):
        """Save the model with GC layer marker."""
        # First handle the standard save
        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            save_function=save_function,
            **kwargs,
        )
        print("HHHH")
        print(is_main_process)
        # Save GC config using the provided save_function
        if is_main_process:
            gc_config = {"is_gc_model": True}
            gc_config_path = os.path.join(save_directory, "gc_config.json")
            save_function(gc_config, gc_config_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Unified loading function for both HF models and custom checkpoints."""
        gc_config_path = os.path.join(pretrained_model_name_or_path, "gc_config.json")
        is_gc_checkpoint = os.path.exists(gc_config_path)

        if is_gc_checkpoint:
            try:
                # Load gc_config to verify it's our format
                gc_config = torch.load(gc_config_path)
                if isinstance(gc_config, dict) and gc_config.get("is_gc_model", False):
                    print("Loading GC checkpoint...")
                    # Direct loading for GC checkpoints
                    model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
                    return model
            except:
                # If there's any error reading gc_config, treat as standard model
                pass

        print("Loading and converting standard model...")
        # For HF models or non-GC checkpoints, handle weight transposition
        kwargs['ignore_mismatched_sizes'] = True
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = cls._replace_layers(model)

        # Fix transposed weights
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            if any(x in name for x in ['c_attn', 'c_fc', 'c_proj']) and 'weight' in name:
                param.data = param.data.t()

        return model

    @classmethod
    def _replace_layers(cls, model: nn.Module) -> nn.Module:
        """Replace both Conv1D and Linear layers."""

        # First, handle Conv1D layers in transformer blocks
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            for i, block in enumerate(model.transformer.h):
                # Replace attention layers
                if hasattr(block, 'attn'):
                    if hasattr(block.attn, 'c_attn'):
                        new_layer = cls._create_gc_layer(block.attn.c_attn)
                        block.attn.c_attn = new_layer

                    if hasattr(block.attn, 'c_proj'):
                        new_layer = cls._create_gc_layer(block.attn.c_proj)
                        block.attn.c_proj = new_layer

                # Replace MLP layers
                if hasattr(block, 'mlp'):
                    if hasattr(block.mlp, 'c_fc'):
                        new_layer = cls._create_gc_layer(block.mlp.c_fc)
                        block.mlp.c_fc = new_layer

                    if hasattr(block.mlp, 'c_proj'):
                        new_layer = cls._create_gc_layer(block.mlp.c_proj)
                        block.mlp.c_proj = new_layer

        # Then recursively replace all Linear layers
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                new_layer = cls._create_gc_layer(module)
                setattr(model, name, new_layer)
            elif isinstance(module, Conv1D):
                new_layer = cls._create_gc_layer(module)
                setattr(model, name, new_layer)
            else:
                # Recursively handle nested modules
                cls._replace_layers(module)

        return model

    @staticmethod
    def _create_gc_layer(old_layer: Union[Conv1D, nn.Linear]) -> nn.Module:
        """Create a new GC layer from either Conv1D or Linear layer."""
        if isinstance(old_layer, Conv1D):
            # Conv1D uses transposed weights compared to Linear
            new_layer = GCLinear(
                in_features=old_layer.weight.shape[0],  # nx
                out_features=old_layer.weight.shape[1], # nf
                bias=old_layer.bias is not None
            )

            # Copy weights with transposition
            new_layer.weight.data = old_layer.weight.t().clone()
            if old_layer.bias is not None:
                new_layer.bias.data = old_layer.bias.clone()

        elif isinstance(old_layer, nn.Linear):
            new_layer = GCLinear(
                in_features=old_layer.in_features,
                out_features=old_layer.out_features,
                bias=old_layer.bias is not None
            )

            new_layer.weight.data = old_layer.weight.clone()
            if old_layer.bias is not None:
                new_layer.bias.data = old_layer.bias.clone()

        else:
            raise ValueError(f"Unsupported layer type: {type(old_layer)}")

        return new_layer

    @classmethod
    def from_config(cls, config, **kwargs):
        """Override from_config to replace layers after initialization."""
        model = super().from_config(config, **kwargs)
        model = cls._replace_layers(model)
        return model