from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import Conv1D
from .layers.linear import GIPLinear, GIPEmbedding
from .layers.layer_norm import GIPLayerNorm
from .helper import transpose_Conv1D
import torch
import torch.nn as nn
import os

class GIPGPT2LMHeadModel(GPT2LMHeadModel):
    """Custom GPT2 model that directly inherits from GPT2LMHeadModel."""
    def __init__(self, config):
        super().__init__(config)
        self.hooks = []
        self._replace_with_GIP_layers()

    # def _register_embedding_hooks(self, token_embedding, position_embedding):
    #     """Register hooks to capture embedding outputs"""

    #     def token_hook(module, input, output):
    #         # Store token embedding output for later use
    #         # token_embedding.token_output = output  # Detach to prevent gradient issues
    #         # token_embedding.pre_activation = output
    #         pass

    #     def position_hook(module, input, output):
    #         # Get the token embeddings and ensure they require grad
    #         if hasattr(token_embedding, 'token_output'):
    #             # token_emb = token_embedding.token_output
    #             # # Create a new tensor that requires grad
    #             # combined = output + token_emb
    #             # position_embedding.pre_activation = combined
    #             # # Store the combined output for gradient computation
    #             # position_embedding.combined_output = combined
    #             # return combined  # Return the combined output to maintain gradient flow
    #             pass
    #         else:
    #             # print("Warning: token_output not found in token_embedding")
    #             # position_embedding.pre_activation = output
    #             # return output
    #             pass

    #     # Register the hooks
    #     token_hook_handle = token_embedding.register_forward_hook(token_hook)
    #     position_hook_handle = position_embedding.register_forward_hook(position_hook)

    #     # Store hook handles for potential cleanup
    #     self.hooks.extend([token_hook_handle, position_hook_handle])

    def _replace_with_GIP_layers(self):
        """
        Replace standard layers with GIP layers in GPT2 model.
        """
        # Replace word embeddings
        if hasattr(self.transformer, 'wte'):
            token_embedding = GIPEmbedding(
                num_embeddings=self.transformer.wte.num_embeddings,
                embedding_dim=self.transformer.wte.embedding_dim
            )
            self.transformer.wte = token_embedding

        if hasattr(self.transformer, 'wpe'):
            position_embedding = GIPEmbedding(
                num_embeddings=self.transformer.wpe.num_embeddings,
                embedding_dim=self.transformer.wpe.embedding_dim
            )
            self.transformer.wpe = position_embedding

        # Replace layers in transformer blocks
        if hasattr(self, 'transformer') and hasattr(self.transformer, 'h'):
            for i, block in enumerate(self.transformer.h):
                # Replace attention layers
                if hasattr(block, 'attn'):
                    if hasattr(block.attn, 'c_attn'):
                        new_layer = self._create_GIPLinear(block.attn.c_attn)
                        block.attn.c_attn = new_layer

                    if hasattr(block.attn, 'c_proj'):
                        new_layer = self._create_GIPLinear(block.attn.c_proj)
                        block.attn.c_proj = new_layer

                # Replace MLP layers
                if hasattr(block, 'mlp'):
                    if hasattr(block.mlp, 'c_fc'):
                        new_layer = self._create_GIPLinear(block.mlp.c_fc)
                        block.mlp.c_fc = new_layer

                    if hasattr(block.mlp, 'c_proj'):
                        new_layer = self._create_GIPLinear(block.mlp.c_proj)
                        block.mlp.c_proj = new_layer

                # Replace LayerNorm layers
                for ln_attr in ['ln_1', 'ln_2']:
                    if hasattr(block, ln_attr):
                        new_ln_layer = self._create_GIPLayernorm(getattr(block, ln_attr))
                        setattr(block, ln_attr, new_ln_layer)

        # Replace final layers
        if hasattr(self, 'lm_head'):
            new_layer = self._create_GIPLinear(self.lm_head)
            self.lm_head = new_layer

        if hasattr(self.transformer, 'ln_f'):
            new_ln_layer = self._create_GIPLayernorm(self.transformer.ln_f)
            self.transformer.ln_f = new_ln_layer

    def _create_GIPLinear(self, old_layer):
        """
        Create a GIP layer from either Conv1D or Linear layer.
        """
        if isinstance(old_layer, Conv1D):
            # Conv1D uses transposed weights compared to Linear
            new_layer = GIPLinear(
                in_features=old_layer.weight.shape[0],  # nx
                out_features=old_layer.weight.shape[1], # nf
                bias=old_layer.bias is not None
            )
            new_layer.weight.data = old_layer.weight.t().clone() # Copy weights with transposition
            if old_layer.bias is not None:
                new_layer.bias.data = old_layer.bias.clone()
        elif isinstance(old_layer, nn.Linear):
            new_layer = GIPLinear(
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

    def _create_GIPLayernorm(self, old_layer):
        """
        Create a GIPLayerNorm from a standard LayerNorm.
        """
        if isinstance(old_layer, nn.LayerNorm):
            # Create the custom LayerNorm layer with the same configuration
            new_layer = GIPLayerNorm(
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
        """
        Load pretrained model and convert weight to be compatible with GIP layers.
        """
        original_model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        state_dict = transpose_Conv1D(original_model.state_dict())
        config = original_model.config
        model = cls(config)
        model.load_state_dict(state_dict, strict=True)
        return model

    def save_pretrained(self, save_directory: str, is_main_process: bool = True, save_function = torch.save, **kwargs):
        """
        Save the GIP model weight while converting back to standard model format (Conv1D).
        """
        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            self.config.save_pretrained(save_directory)
            state_dict = self.state_dict()
            converted_state_dict = transpose_Conv1D(state_dict)
            from safetensors.torch import save_file
            save_file(converted_state_dict, os.path.join(save_directory, "model.safetensors"))

    def __del__(self):
        """Clean up hooks when the model is deleted"""
        for hook in self.hooks:
            hook.remove()