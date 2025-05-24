from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Any
if TYPE_CHECKING:
    from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from .MLPHook import HookManager

class MLPGradientExtractor:
    """
    Extracts raw gradients and pre-activations from model layers using hooks,
    without requiring custom layer implementations or projections.
    Processes one layer at a time to save memory with optional CPU offloading.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        cpu_offload: bool = False,
    ) -> None:
        """
        Initialize the gradient extractor.

        Args:
            model (nn.Module): PyTorch model.
            device (str): Device to run the model on.
            cpu_offload (bool): Whether to offload data to CPU to save GPU memory.
        """
        self.model = model
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.cpu_offload = cpu_offload

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
            self.cpu_offload
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

        # Clear GPU cache after processing training data
        torch.cuda.empty_cache()

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

        # Clear GPU cache after processing
        torch.cuda.empty_cache()

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
        # Initialize lists to collect gradients, explicitly on CPU
        pre_activations = []
        input_features = []

        # Store tensor dimensions for later use
        tensor_dims = None
        is_3d = None

        # Get the layer from the model to check if it has bias
        layer = dict(self.model.named_modules())[layer_name]
        # Check if the layer has bias (works for nn.Linear, nn.Conv2d, etc.)
        has_bias = hasattr(layer, 'bias') and layer.bias is not None

        # Process each batch
        for batch_idx, batch in tqdm(enumerate(dataloader), desc=f"Processing {dataset_type} data", total=len(dataloader)):
            try:
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

                        # Store dimensions for later use if not already stored
                        if tensor_dims is None:
                            is_3d = pre_act_grad.dim() == 3
                            tensor_dims = {
                                'pre_act_shape': pre_act_grad.shape,
                                'input_feat_shape': input_feat.shape,
                                'dtype': pre_act_grad.dtype
                            }

                        # EXPLICITLY force tensors to CPU and detach them
                        if self.cpu_offload:
                            # Clone first to make sure we have a new tensor that doesn't share storage
                            pre_act_cpu = pre_act_grad.detach().clone().cpu()
                            input_feat_cpu = input_feat.detach().clone().cpu()

                            # Append CPU tensors
                            pre_activations.append(pre_act_cpu)
                            input_features.append(input_feat_cpu)

                            # Explicitly delete GPU tensors
                            del pre_act_grad, input_feat, pre_act_cpu, input_feat_cpu
                        else:
                            # If not offloading, still detach from computation graph
                            pre_activations.append(pre_act_grad.detach().clone())
                            input_features.append(input_feat.detach().clone())
                            del pre_act_grad, input_feat

                    # Clear hook data explicitly
                    if hasattr(hook_manager, 'clear_data'):
                        hook_manager.clear_data(layer_name)

                    # Clear all local variables from this iteration
                    if 'inputs' in locals(): del inputs
                    if 'outputs' in locals(): del outputs
                    if 'loss' in locals(): del loss
                    if 'labels' in locals(): del labels
                    if 'comp_data' in locals(): del comp_data

                    # Clear GPU cache
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                # Clean up on error
                torch.cuda.empty_cache()
                continue

        # Process collected gradients
        if not pre_activations or not input_features:
            return None

        print(f"Concatenating {len(pre_activations)} batches of data...")

        # Concatenate all batches while keeping them on CPU
        try:
            if self.cpu_offload:
                with torch.no_grad():
                    # Keep tensors on CPU during concatenation
                    pre_act_tensor = torch.cat(pre_activations, dim=0)
                    input_feat_tensor = torch.cat(input_features, dim=0)

                    # Clear original lists to free memory
                    del pre_activations
                    del input_features
                    pre_activations = []
                    input_features = []
            else:
                with torch.no_grad():
                    pre_act_tensor = torch.cat(pre_activations, dim=0).to(self.device)
                    input_feat_tensor = torch.cat(input_features, dim=0).to(self.device)

                    # Clear original lists to free memory
                    del pre_activations
                    del input_features
                    pre_activations = []
                    input_features = []

            # Get batch size from the tensor shape before potentially adding bias column
            batch_size = input_feat_tensor.shape[0]

            # Processing steps that must happen on device
            if self.cpu_offload:
                # Create final tensors directly on CPU with minimal GPU usage
                with torch.no_grad():
                    # Scale by batch size for per-sample gradients (do this on CPU)
                    pre_act_tensor = pre_act_tensor * batch_size

                    # Only add bias term if the layer has bias (do this on CPU)
                    if has_bias:
                        if is_3d:
                            # For 3D tensors (batch_size, seq_length, features)
                            batch_size, seq_length, hidden_size = input_feat_tensor.shape

                            ones = torch.ones(
                                batch_size, seq_length, 1,
                                device='cpu',  # Keep on CPU
                                dtype=input_feat_tensor.dtype
                            )
                            input_feat_tensor = torch.cat([input_feat_tensor, ones], dim=2)
                            del ones
                        else:
                            # For 2D tensors (batch_size, features)
                            ones = torch.ones(
                                batch_size, 1,
                                device='cpu',  # Keep on CPU
                                dtype=input_feat_tensor.dtype
                            )
                            input_feat_tensor = torch.cat([input_feat_tensor, ones], dim=1)
                            del ones
            else:
                with torch.no_grad():
                    # Scale by batch size for per-sample gradients
                    pre_act_tensor = pre_act_tensor * batch_size

                    # Only add bias term if the layer has bias
                    if has_bias:
                        if is_3d:
                            # For 3D tensors (batch_size, seq_length, features)
                            batch_size, seq_length, hidden_size = input_feat_tensor.shape

                            ones = torch.ones(
                                batch_size, seq_length, 1,
                                device=input_feat_tensor.device,
                                dtype=input_feat_tensor.dtype
                            )
                            input_feat_tensor = torch.cat([input_feat_tensor, ones], dim=2)
                            del ones
                        else:
                            # For 2D tensors (batch_size, features)
                            ones = torch.ones(
                                batch_size, 1,
                                device=input_feat_tensor.device,
                                dtype=input_feat_tensor.dtype
                            )
                            input_feat_tensor = torch.cat([input_feat_tensor, ones], dim=1)
                            del ones

            # Clear GPU cache
            torch.cuda.empty_cache()

            return {
                'pre_activation': pre_act_tensor,
                'input_features': input_feat_tensor,
                'is_3d': is_3d,
            }

        except Exception as e:
            print(f"Error during final tensor processing: {str(e)}")
            # Clean up on error
            if 'pre_act_tensor' in locals(): del pre_act_tensor
            if 'input_feat_tensor' in locals(): del input_feat_tensor
            torch.cuda.empty_cache()
            return None