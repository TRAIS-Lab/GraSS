from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Any, Union
if TYPE_CHECKING:
    from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

class GradientExtractor:
    """
    Extracts raw gradients from the entire model directly,
    without requiring hooks or layer-by-layer processing.
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
        self.model.eval()  # Set model to evaluation mode
        self.device = device
        self.cpu_offload = cpu_offload

        # Calculate total number of parameters
        self.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {self.total_params} trainable parameters")

    def extract_gradients(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        custom_loss_fn: Optional[callable] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Extract gradients from the entire model for both training and test data.

        Args:
            train_dataloader: DataLoader containing training data
            test_dataloader: DataLoader containing test data
            custom_loss_fn: Optional custom loss function to use instead of default

        Returns:
            Tuple of (training gradients, test gradients)
        """
        print("Processing training data...")
        train_gradients = self._process_dataloader(
            train_dataloader,
            "training",
            custom_loss_fn
        )

        # Clear GPU cache after processing training data
        torch.cuda.empty_cache()

        print("Processing test data...")
        test_gradients = self._process_dataloader(
            test_dataloader,
            "test",
            custom_loss_fn
        )

        # Clear GPU cache after processing
        torch.cuda.empty_cache()

        return train_gradients, test_gradients

    def _extract_flat_gradients(self) -> torch.Tensor:
        """
        Extract flattened gradients from the entire model.

        Returns:
            Flattened gradient tensor
        """
        # Collect gradients from all parameters
        grads = []
        for param in self.model.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grads.append(param.grad.detach().view(-1))
                else:
                    # If gradient is None (possible if some layers don't receive gradients),
                    # append zeros
                    grads.append(torch.zeros_like(param).view(-1))

        # Concatenate all gradients
        if not grads:
            return None

        flat_grad = torch.cat(grads)

        # Offload to CPU if needed
        return flat_grad.cpu() if self.cpu_offload else flat_grad

    def _process_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataset_type: str,
        custom_loss_fn: Optional[callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a dataloader and extract gradients for the entire model.

        Args:
            dataloader: DataLoader to process
            dataset_type: Either "training" or "test" (for logging)
            custom_loss_fn: Optional custom loss function

        Returns:
            Dictionary with processed gradients
        """
        # Initialize list to collect gradients
        gradients = []

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
                        loss = outputs.loss
                    else:
                        # Fallback to standard cross-entropy if no loss in outputs
                        if 'labels' in locals():
                            loss = nn.functional.cross_entropy(outputs, labels)
                        else:
                            raise ValueError("No labels provided and model doesn't return loss.")

                # Backward pass
                loss.backward()

                # Get gradients
                with torch.no_grad():
                    # Extract flat gradients from the entire model
                    flat_grad = self._extract_flat_gradients()

                    if flat_grad is not None:
                        # Make sure to detach and clone
                        if self.cpu_offload:
                            grad_cpu = flat_grad.detach().clone().cpu()
                            gradients.append(grad_cpu)
                            del flat_grad, grad_cpu
                        else:
                            gradients.append(flat_grad.detach().clone())
                            del flat_grad

                    # Clear all local variables from this iteration
                    if 'inputs' in locals(): del inputs
                    if 'outputs' in locals(): del outputs
                    if 'loss' in locals(): del loss
                    if 'labels' in locals(): del labels

                # Clear GPU cache
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                # Clean up on error
                torch.cuda.empty_cache()
                continue

        # Process collected gradients
        if not gradients:
            return None

        print(f"Concatenating {len(gradients)} batches of data...")

        # Stack gradients to create tensor of shape [batch_size, total_params]
        try:
            if self.cpu_offload:
                with torch.no_grad():
                    # Keep tensors on CPU during stacking
                    gradient_tensor = torch.stack(gradients, dim=0)

                    # Clear original list to free memory
                    del gradients
                    gradients = []
            else:
                with torch.no_grad():
                    gradient_tensor = torch.stack(gradients, dim=0).to(self.device)

                    # Clear original list to free memory
                    del gradients
                    gradients = []

            # Get batch size from the tensor shape
            batch_size = gradient_tensor.shape[0]

            # Scale by batch size for per-sample gradients (optional)
            with torch.no_grad():
                gradient_tensor = gradient_tensor * batch_size

            # Clear GPU cache
            torch.cuda.empty_cache()

            return {
                'gradient': gradient_tensor,
                'total_params': self.total_params
            }

        except Exception as e:
            print(f"Error during final tensor processing: {str(e)}")
            # Clean up on error
            if 'gradient_tensor' in locals(): del gradient_tensor
            torch.cuda.empty_cache()
            return None

    def get_param_to_indices_map(self):
        """
        Get a mapping from parameter names to their indices in the flattened gradient.

        Returns:
            Dictionary mapping parameter names to (start_idx, end_idx) tuples
        """
        param_to_indices = {}
        current_idx = 0

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_size = param.numel()
                param_to_indices[name] = (current_idx, current_idx + param_size)
                current_idx += param_size

        return param_to_indices