import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '...'))
sys.path.append(parent_dir)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import Conv1D
from _gradcomp.layers.linear import GCLinear, GCEmbedding
from _gradcomp.layers.layer_norm import GCLayerNorm
from GC.utlis import transpose_Conv1D
import torch
import torch.nn as nn
import os

class GCGPT2LMHeadModel(GPT2LMHeadModel):
    """
    Custom GPT2 model that directly inherits from GPT2LMHeadModel with Gradient Component (GC) layers.
    """
    def __init__(self, config):
        super().__init__(config)
        self.hooks = []
        self._replace_with_GC_layers()


    def _replace_with_GC_layers(self):
        """
        Replace standard layers with GC layers in GPT2 model.
        """
        # Replace word embeddings
        if hasattr(self.transformer, 'wte'):
            token_embedding = GCEmbedding(
                num_embeddings=self.transformer.wte.num_embeddings,
                embedding_dim=self.transformer.wte.embedding_dim
            )
            self.transformer.wte = token_embedding

        if hasattr(self.transformer, 'wpe'):
            position_embedding = GCEmbedding(
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
                        new_layer = self._create_GCLinear(block.attn.c_attn)
                        block.attn.c_attn = new_layer

                    if hasattr(block.attn, 'c_proj'):
                        new_layer = self._create_GCLinear(block.attn.c_proj)
                        block.attn.c_proj = new_layer

                # Replace MLP layers
                if hasattr(block, 'mlp'):
                    if hasattr(block.mlp, 'c_fc'):
                        new_layer = self._create_GCLinear(block.mlp.c_fc)
                        block.mlp.c_fc = new_layer

                    if hasattr(block.mlp, 'c_proj'):
                        new_layer = self._create_GCLinear(block.mlp.c_proj)
                        block.mlp.c_proj = new_layer

                # Replace LayerNorm layers
                for ln_attr in ['ln_1', 'ln_2']:
                    if hasattr(block, ln_attr):
                        new_ln_layer = self._create_GCLayerNorm(getattr(block, ln_attr))
                        setattr(block, ln_attr, new_ln_layer)

        # Replace final layers
        if hasattr(self, 'lm_head'):
            new_layer = self._create_GCLinear(self.lm_head)
            self.lm_head = new_layer

        if hasattr(self.transformer, 'ln_f'):
            new_ln_layer = self._create_GCLayerNorm(self.transformer.ln_f)
            self.transformer.ln_f = new_ln_layer

    def _create_GCLinear(self, old_layer):
        """
        Create a GC layer from either Conv1D or Linear layer.
        """
        if isinstance(old_layer, Conv1D):
            # Conv1D uses transposed weights compared to Linear
            new_layer = GCLinear(
                in_features=old_layer.weight.shape[0],  # nx
                out_features=old_layer.weight.shape[1], # nf
                bias=old_layer.bias is not None
            )
            new_layer.weight.data = old_layer.weight.t().clone() # Copy weights with transposition
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

    def _create_GCLayerNorm(self, old_layer):
        """
        Create a GCLayerNorm from a standard LayerNorm.
        """
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

    def set_projectors(self, projector_kwargs, train_dataloader):
        """
        Set projectors for all GC layers in the model.

        Args:
            projector_kwargs: Dictionary containing projector configuration.
            train_dataloader: Dataloader for training data. Used to get the input shape for the first layer.
        """
        if projector_kwargs is None:
            return

        for batch in train_dataloader:
            self.forward(batch["input_ids"].cuda(self.device), attention_mask=batch["attention_mask"].cuda(self.device))
            break

        proj_seed = projector_kwargs.get('proj_seed', 0)
        proj_dim = projector_kwargs.get("proj_dim", 32)
        proj_dim_dist = projector_kwargs.get("proj_dim_dist", "uniform")

        projector_kwargs.pop("proj_seed")
        projector_kwargs.pop("proj_dim")
        projector_kwargs.pop("proj_dim_dist")

        # Control the thresholding for different vals
        projector_kwargs_1 = projector_kwargs.copy()
        projector_kwargs_2 = projector_kwargs.copy()

        # projector_kwargs_2["threshold"] = 0.0 #TODO threshold for gradient of pre-activation, which shouldn't be sparse so better set to 0

        layer_dim_1 = []
        layer_dim_2 = []
        for layer_id, layer in enumerate(self.modules()):
            if isinstance(layer, GCLinear):
                layer_dim_1.append(layer.weight.shape[0])
                layer_dim_2.append(layer.weight.shape[0] * layer.weight.shape[1])
            elif isinstance(layer, GCEmbedding):
                layer_dim_1.append(layer.embedding_dim)
                layer_dim_2.append(layer.embedding_dim * layer.num_embeddings)
            elif isinstance(layer, GCLayerNorm):
                layer_dim_1.append(layer.normalized_shape[0])
                layer_dim_2.append(layer.normalized_shape[0])
            else:
                layer_dim_1.append(0)
                layer_dim_2.append(0)

        if proj_dim_dist == "NU": # Non-uniform projection dimension
            total_dim_1 = sum(layer_dim_1)
            total_dim_2 = sum(layer_dim_2)
            proj_dim_1 = [int(proj_dim * dim / total_dim_1) for dim in layer_dim_1]
            proj_dim_2 = [int(proj_dim * dim / total_dim_2) for dim in layer_dim_2]
        elif proj_dim_dist == "U": # Uniform projection dimension
            proj_dim_1 = [proj_dim] * len(layer_dim_1)
            proj_dim_2 = [proj_dim] * len(layer_dim_2)

        # Apply projectors to all GC layers
        for module_id, module in enumerate(self.modules()):
            if isinstance(module, GCLinear) or isinstance(module, GCEmbedding) or isinstance(module, GCLayerNorm):
                base_seed = proj_seed + int(1e4) * module_id
                module.set_projector(base_seed, proj_dim_1[module_id], proj_dim_2[module_id], projector_kwargs_1, projector_kwargs_2)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load pretrained model and convert weight to be compatible with GC layers.
        """
        original_model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        state_dict = transpose_Conv1D(original_model.state_dict())
        config = original_model.config
        model = cls(config)
        model.load_state_dict(state_dict, strict=True)
        return model

    def save_pretrained(self, save_directory: str, is_main_process: bool = True, save_function = torch.save, **kwargs):
        """
        Save the GC model weight while converting back to standard model format (Conv1D).
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