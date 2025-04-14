from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Any
if TYPE_CHECKING:
    from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from .hook import HookManager
import gc

class GradientExtractor:
    """
    Extracts raw gradients and pre-activations from model layers using hooks,
    without requiring custom layer implementations or projections.
    Processes one layer at a time to save memory.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
    ) -> None:
        """
        Initialize the gradient extractor.

        Args:
            model (nn.Module): PyTorch model.
            device (str): Device to run the model on.
            cpu_offload (bool): Whether to offload data to CPU to save GPU memory.
            profile (bool): Whether to record timing information.
        """
        self.model = model
        self.model.to(device)
        self.model.eval()

        self.device = device

    def extract_gradients_for_layer(
        self,
        layer_name: str,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        custom_loss_fn: Optional[callable] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Extract gradients for a specific layer from both training and test data.
        Processes one layer at a time to save memory.

        Args:
            layer_name: Name of the layer to extract gradients from
            train_dataloader: DataLoader containing training data
            test_dataloader: DataLoader containing test data
            custom_loss_fn: Optional custom loss function to use instead of default

        Returns:
            Tuple of (training components, test components), each with keys:
                - 'pre_activation': Processed pre-activation gradients
                - 'input_features': Processed input features
        """
        # Create hook manager for just this layer
        hook_manager = HookManager(
            self.model,
            [layer_name],
        )

        print(f"Processing layer: {layer_name}")
        # First process training data
        train_components = self._process_dataloader(
            hook_manager,
            train_dataloader,
            layer_name,
            "training",
            custom_loss_fn
        )

        # Then process test data
        test_components = self._process_dataloader(
            hook_manager,
            test_dataloader,
            layer_name,
            "test",
            custom_loss_fn
        )

        # Remove hooks after processing both datasets
        hook_manager.remove_hooks()

        return train_components, test_components

    def _process_dataloader(
        self,
        hook_manager: HookManager,
        dataloader: torch.utils.data.DataLoader,
        layer_name: str,
        dataset_type: str,
        custom_loss_fn: Optional[callable] = None,
    ) -> Dict[str, Tensor]:
        """
        Process a dataloader and extract gradients for the specified layer.

        Args:
            hook_manager: HookManager instance
            dataloader: DataLoader to process
            layer_name: Name of the layer to extract gradients from
            dataset_type: Either "training" or "test" (for logging)
            custom_loss_fn: Optional custom loss function

        Returns:
            Dictionary with processed gradient components
        """
        # Initialize lists to collect gradients, offloaded to CPU
        pre_activations = []
        input_features = []

        # Process each batch
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {dataset_type} data")):
            # Zero gradients
            self.model.zero_grad()

            # Prepare inputs
            if isinstance(batch, dict):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
            else:
                inputs = batch[0].to(self.device)
                if len(batch) > 1:
                    labels = batch[1].to(self.device)

            # Forward pass
            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

            # Compute loss
            if custom_loss_fn:
                loss = custom_loss_fn(outputs, batch)
            else:
                # Default loss calculation (assuming model returns loss)
                if hasattr(outputs, 'loss'):
                    logp = -outputs.loss
                    loss = logp - torch.log(1 - torch.exp(logp))
                else:
                    # Fallback to standard cross-entropy if no loss in outputs
                    if 'labels' in locals():
                        loss = nn.functional.cross_entropy(outputs, labels)
                    else:
                        raise ValueError("No labels provided and model doesn't return loss.")

            # Backward pass
            loss.backward()

            # Get gradients from hook manager
            with torch.no_grad():
                # Get raw gradients for this layer
                comp_data = hook_manager.get_gradient_components(layer_name)
                if comp_data:
                    pre_act_grad, input_feat = comp_data
                    pre_activations.append(pre_act_grad.detach())
                    input_features.append(input_feat.detach())

        # Process collected gradients
        if not pre_activations or not input_features:
            return None

        # Concatenate all batches
        pre_act_tensor = torch.cat(pre_activations, dim=0).to(self.device)
        input_feat_tensor = torch.cat(input_features, dim=0).to(self.device)

        # Determine if tensors are 3D (sequence models) or 2D
        is_3d = pre_act_tensor.dim() == 3

        # Handle bias term for linear layers by adding a column of ones
        if is_3d:
            # For 3D tensors (batch_size, seq_length, features)
            batch_size, seq_length, hidden_size = input_feat_tensor.shape
            ones = torch.ones(
                batch_size, seq_length, 1,
                device=self.device,
                dtype=input_feat_tensor.dtype
            )
            input_feat_tensor = torch.cat([input_feat_tensor, ones], dim=2)
        else:
            # For 2D tensors (batch_size, features)
            batch_size = input_feat_tensor.shape[0]
            ones = torch.ones(
                batch_size, 1,
                device=self.device,
                dtype=input_feat_tensor.dtype
            )
            input_feat_tensor = torch.cat([input_feat_tensor, ones], dim=1)

        # Scale by batch size for per-sample gradients
        pre_act_tensor = pre_act_tensor * batch_size

        # Clean up individual batch tensors to save memory
        del pre_activations
        del input_features
        gc.collect()

        return {
            'pre_activation': pre_act_tensor,
            'input_features': input_feat_tensor,
            'is_3d': is_3d
        }