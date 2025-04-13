import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '...'))
sys.path.append(parent_dir)

from transformers import LlamaForCausalLM
from _GradComp.layers.linear import GCLinear, GCEmbedding
from _GradComp.layers.layer_norm import GCLayerNorm
import torch
import torch.nn as nn
import os

class GCLlamaForCausalLM(LlamaForCausalLM):
    """
    Custom Llama 3.2 model that inherits from LlamaForCausalLM with Gradient Component (GC) layers.
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
            if isinstance(child, nn.Linear):
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
        Create a GC layer from a Linear layer.
        """
        assert isinstance(old_layer, nn.Linear), "Layer must be Linear"
        new_layer = GCLinear(
            in_features=old_layer.in_features,
            out_features=old_layer.out_features,
            bias=old_layer.bias is not None
        )
        new_layer.weight.data = old_layer.weight.clone()
        if old_layer.bias is not None:
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

        # Get a batch of training data to initialize the model
        train_batch = next(iter(train_dataloader))
        self.forward(
            input_ids=train_batch["input_ids"].to(self.device),
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

                # Handle active indices for Localize method
                if projector_kwargs.get("method") == "Localize":
                    active_indices = None
                    try:
                        dim = projector_kwargs["proj_dim"]
                        if proj_factorize:
                            mask_path = f"../Llama3_OWT/Localize/mask_{dim}*{dim}/{module_name}.pt"
                        else:
                            mask_path = f"../Llama3_OWT/Localize/mask_{dim}/{module_name}.pt"
                        active_indices = torch.load(mask_path, weights_only=False)
                    except FileNotFoundError:
                        print(f"Mask file not found for {module_name}. Using default active indices.")

                    projector_kwargs["active_indices"] = active_indices
                else:
                    projector_kwargs["active_indices"] = None

                module.set_projector(base_seed, projector_kwargs, proj_factorize)

    def __del__(self):
        """Clean up hooks when the model is deleted"""
        for hook in self.hooks:
            hook.remove()