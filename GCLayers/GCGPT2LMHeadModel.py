from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import Conv1D
from .linear import GCLinear, GCEmbedding
from .layer_norm import GCLayerNorm
import torch
import torch.nn as nn
import os
import json

class GCGPT2LMHeadModel(GPT2LMHeadModel):
    """Custom GPT2 model that directly inherits from GPT2LMHeadModel."""
    def __init__(self, config):
        super().__init__(config)
        self._replace_with_gc_layers()

    def _replace_with_gc_layers(self):
        """Replace standard layers with GC layers in GPT2 model."""
        # Replace embedding layers
        if hasattr(self.transformer, 'wte'):
            self.transformer.wte = GCEmbedding(
                num_embeddings=self.transformer.wte.num_embeddings,
                embedding_dim=self.transformer.wte.embedding_dim
            )

        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe = GCEmbedding(
                num_embeddings=self.transformer.wpe.num_embeddings,
                embedding_dim=self.transformer.wpe.embedding_dim
            )

        # Replace layers in transformer blocks
        if hasattr(self, 'transformer') and hasattr(self.transformer, 'h'):
            for i, block in enumerate(self.transformer.h):
                # Replace attention layers
                if hasattr(block, 'attn'):
                    if hasattr(block.attn, 'c_attn'):
                        new_layer = self._create_gc_layer(block.attn.c_attn)
                        block.attn.c_attn = new_layer

                    if hasattr(block.attn, 'c_proj'):
                        new_layer = self._create_gc_layer(block.attn.c_proj)
                        block.attn.c_proj = new_layer

                # Replace MLP layers
                if hasattr(block, 'mlp'):
                    if hasattr(block.mlp, 'c_fc'):
                        new_layer = self._create_gc_layer(block.mlp.c_fc)
                        block.mlp.c_fc = new_layer

                    if hasattr(block.mlp, 'c_proj'):
                        new_layer = self._create_gc_layer(block.mlp.c_proj)
                        block.mlp.c_proj = new_layer

                    # Replace LayerNorm layers (e.g., ln_1, ln_2)
                for ln_attr in ['ln_1', 'ln_2']:
                    if hasattr(block, ln_attr):
                        new_ln_layer = self._create_gc_layernorm(getattr(block, ln_attr))
                        setattr(block, ln_attr, new_ln_layer)

        # Replace lm_head if it exists
        if hasattr(self, 'lm_head'):
            new_layer = self._create_gc_layer(self.lm_head)
            self.lm_head = new_layer

        # Replace the final layer normalization
        if hasattr(self.transformer, 'ln_f'):
            new_ln_layer = self._create_gc_layernorm(self.transformer.ln_f)
            self.transformer.ln_f = new_ln_layer

    def _create_gc_layer(self, old_layer):
        """Create a GC layer from either Conv1D or Linear layer."""
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

    def _create_gc_layernorm(self, old_layer):
        """Create a GCLayerNorm from a standard LayerNorm."""
        if isinstance(old_layer, nn.LayerNorm):
            # Create the custom LayerNorm layer with the same configuration
            new_layer = GCLayerNorm(
                normalized_shape=old_layer.normalized_shape,
                elementwise_affine=old_layer.elementwise_affine,
            )
            # Copy weights and biases if they exist
            if old_layer.elementwise_affine:
                new_layer.weight.data = old_layer.weight.clone()
                new_layer.bias.data = old_layer.bias.clone()
        else:
            raise ValueError(f"Unsupported layer type for LayerNorm: {type(old_layer)}")
        return new_layer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load pretrained model, handling GC layer conversion efficiently."""
        # Check if this is already a GC checkpoint
        gc_config_path = os.path.join(pretrained_model_name_or_path, "gc_config.json")
        is_gc_checkpoint = os.path.exists(gc_config_path)

        if is_gc_checkpoint:
            print("Loading GC checkpoint directly...")
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        print("Converting standard checkpoint to GC model...")
        # First load the original model
        original_model = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        # Create our GC model
        config = original_model.config
        model = cls(config)

        # Handle weight loading with custom state dict preparation
        state_dict = original_model.state_dict()
        new_state_dict = {}

        for key, value in state_dict.items():
            # Special handling for weight tensors in specific layers
            if any(x in key for x in ['c_attn', 'c_fc', 'c_proj']) and 'weight' in key:
                # For these layers, we transpose the weight tensor
                new_state_dict[key] = value.t()
            else:
                new_state_dict[key] = value

        # Load the processed state dict
        model.load_state_dict(new_state_dict, strict=False)

        return model

    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        save_function = torch.save,
        **kwargs,
    ):
        """Save model with proper weight transposition."""
        print("Saving GC model...")
        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            with open(os.path.join(save_directory, "gc_config.json"), 'w') as f: json.dump({"is_gc_model": True}, f)
        super().save_pretrained(save_directory, is_main_process=is_main_process, save_function=save_function, **kwargs)