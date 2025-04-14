import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '...'))
sys.path.append(parent_dir)

from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import Conv1D
from _GradComp.layers.linear import GCLinear, GCEmbedding
from _GradComp.layers.layer_norm import GCLayerNorm
from GPT2_wikitext.GradComp.utils import transpose_Conv1D
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
        self._replace_with_GC_layers(self)

    def _replace_with_GC_layers(self, module):
        """
        Recursively traverse model structure and replace layers based on their type.

        Args:
            module: Current module to process
        """
        # Get all named children (direct submodules)
        for name, child in list(module.named_children()):
            # First recursively process any children of this child
            self._replace_with_GC_layers(child)

            # Then check if this child needs replacement
            if isinstance(child, nn.Linear) or isinstance(child, Conv1D):
                # Replace with GCLinear
                new_layer = self._create_GCLinear(child)
                setattr(module, name, new_layer)

            elif isinstance(child, nn.Embedding):
                # Replace with GCEmbedding
                new_layer = GCEmbedding(
                    num_embeddings=child.num_embeddings,
                    embedding_dim=child.embedding_dim,
                    padding_idx=child.padding_idx
                )
                # Copy the weights
                new_layer.weight.data = child.weight.data.clone()
                setattr(module, name, new_layer)

            elif isinstance(child, nn.LayerNorm):
                # Replace with GCLayerNorm
                new_layer = GCLayerNorm(
                    normalized_shape=child.normalized_shape,
                    elementwise_affine=child.elementwise_affine,
                    eps=child.eps
                )
                # Copy weights and biases if they exist
                if child.elementwise_affine:
                    new_layer.weight.data = child.weight.data.clone()
                    new_layer.bias.data = child.bias.data.clone()
                setattr(module, name, new_layer)

    def _create_GCLinear(self, old_layer):
        """
        Create a GC layer from either Conv1D or Linear layer.
        """
        assert isinstance(old_layer, (nn.Linear, Conv1D)), "Layer must be either Linear or Conv1D"
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

        return new_layer

    def _create_GCLayerNorm(self, old_layer):
        """
        Create a GCLayerNorm from a standard LayerNorm.
        """
        assert isinstance(old_layer, nn.LayerNorm), "Layer must be a LayerNorm"
        new_layer = GCLayerNorm(
            normalized_shape=old_layer.normalized_shape,
            elementwise_affine=old_layer.elementwise_affine,
        )
        # Copy weights and biases if they exist
        if old_layer.elementwise_affine:
            new_layer.weight.data = old_layer.weight.clone()
            new_layer.bias.data = old_layer.bias.clone()

        return new_layer

    def set_projectors(self, layer_names, projector_kwargs, train_dataloader):
        """
        Set projectors for all GC layers in the model.

        Args:
            layer_names: Layer's names to set projectors
            projector_kwargs: Dictionary containing projector configuration.
            train_dataloader: Dataloader for training data. Used to get the input shape for the first layer.
        """
        if projector_kwargs is None:
            return

        train_batch = next(iter(train_dataloader))
        self.forward(
            train_batch["input_ids"].to(self.device),
            attention_mask=train_batch["attention_mask"].to(self.device)
        )

        proj_seed = projector_kwargs.get('proj_seed', 0)
        proj_factorize = projector_kwargs.get("proj_factorize", True)

        # Remove these keys as they're handled separately
        if 'proj_seed' in projector_kwargs:
            projector_kwargs.pop("proj_seed")
        if 'proj_factorize' in projector_kwargs:
            projector_kwargs.pop("proj_factorize")

        # Apply projectors to all GC layers
        for module_id, (module_name, module) in enumerate(self.named_modules()):
            if module_name in layer_names:
                base_seed = proj_seed + int(1e4) * module_id
                print(f"Setting projector for {module_name}...")
                # push module_name to projector_kwargs
                if projector_kwargs.get("method") == "Localize":
                    active_indices = None
                    try:
                        dim = projector_kwargs["proj_dim"]
                        if proj_factorize:
                            mask_path = f"../GPT2_wikitext/Localize/mask_{dim}*{dim}/{module_name}.pt"
                        else:
                            mask_path = f"../GPT2_wikitext/Localize/mask_{dim}/{module_name}.pt"
                        active_indices = torch.load(mask_path, weights_only=False)
                    except FileNotFoundError:
                        print(f"Mask file not found for {module_name}. Using default active indices.")

                    projector_kwargs["active_indices"] = active_indices
                else:
                    projector_kwargs["active_indices"] = None
                module.set_projector(base_seed, projector_kwargs, proj_factorize)


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