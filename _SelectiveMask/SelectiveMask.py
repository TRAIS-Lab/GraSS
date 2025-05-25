import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr
import math
from torch.utils.data import DataLoader, TensorDataset

class SelectiveMask:
    def __init__(
            self,
            gradient_dim,
            lambda_reg=1e-1,
            lr=0.01,
            min_active_gradient=10,
            max_active_gradient=None,
            initial_temperature=1.0,
            min_temperature=0.01,
            temp_decay_rate=0, # 1e-4
            device='cpu',
            logger=None
        ):
        """
        Initialize the gradient mask optimizer.

        Args:
            gradient_dim: Dimensionality of the gradient vectors
            lambda_reg: Regularization parameter for sparsity
            lr: Learning rate for optimizer
            min_active_gradient: Minimum number of gradient components that must remain active
            max_active_gradient: Maximum gradient components allowed
            initial_temperature: Starting temperature for sigmoid annealing
            min_temperature: Minimum temperature value
            temp_decay_rate: Rate at which temperature decays per epoch
            device: Device to run computations on
            logger: Logger object for output messages
        """
        self.device = device
        self.lambda_reg = lambda_reg
        self.logger = logger

        # Dimensions
        self.gradient_dim = gradient_dim

        # Constraints for the mask
        self.min_active_gradient = min_active_gradient
        self.max_active_gradient = max_active_gradient if max_active_gradient is not None else 2 * min_active_gradient

        # Temperature annealing parameters
        self.temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.temp_decay_rate = temp_decay_rate

        # Initialize mask parameters for gradient
        self.S_grad = nn.Parameter(torch.randn(gradient_dim, device=device) * 1e-2)

        # Use Adam optimizer for the mask
        self.optimizer = optim.Adam([self.S_grad], lr=lr)

    def _log(self, message, level="info"):
        """Helper method to handle logging"""
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)
        else:
            print(message)

    def sigmoid_grad(self):
        """Return sigmoid(S_grad/T) as the gradient mask with temperature T"""
        return torch.sigmoid(self.S_grad / self.temperature)

    def compute_batch_inner_products(self, test_grad_batch, train_grad_batch, apply_mask=False):
        """
        Compute inner products between test and training gradients for a batch.

        Args:
            test_grad_batch: Batch of test gradients [batch_size, gradient_dim]
            train_grad_batch: Batch of training gradients [batch_size, gradient_dim]
            apply_mask: Whether to apply mask to gradients

        Returns:
            Tensor of inner products [batch_size_test, batch_size_train]
        """
        if apply_mask:
            # Apply mask to gradients
            mask_grad = self.sigmoid_grad()
            test_grad_batch = test_grad_batch * mask_grad
            train_grad_batch = train_grad_batch * mask_grad

        # Compute inner products using matrix multiplication
        return torch.matmul(test_grad_batch, train_grad_batch.T)

    def compute_correlations(self, original_ips_batch, masked_ips_batch):
        """Enhanced correlation computation with better numerical stability"""
        # Cast to float32 for better numerical stability
        original_ips_batch = original_ips_batch.float()
        masked_ips_batch = masked_ips_batch.float()

        # Check for NaN values before processing
        if torch.isnan(original_ips_batch).any() or torch.isnan(masked_ips_batch).any():
            self._log("Warning: NaN values detected in input tensors", level="warning")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Mean center both sets of inner products along train dimension
        orig_centered = original_ips_batch - original_ips_batch.mean(dim=1, keepdim=True)
        masked_centered = masked_ips_batch - masked_ips_batch.mean(dim=1, keepdim=True)

        # Compute variance with larger epsilon
        orig_var = torch.sum(orig_centered**2, dim=1)
        masked_var = torch.sum(masked_centered**2, dim=1)

        # Use a larger epsilon for numerical stability
        epsilon = 1e-6

        # Create a mask for valid samples (non-zero variance)
        valid_samples = (orig_var > epsilon) & (masked_var > epsilon)

        # If no valid samples, return a default loss
        if not torch.any(valid_samples):
            self._log("Warning: No valid samples for correlation", level="warning")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Compute correlation only for valid samples
        numerator = torch.sum(orig_centered * masked_centered, dim=1)
        denominator = torch.sqrt(orig_var * masked_var + epsilon)

        # Compute correlations with safety checks
        correlations = torch.zeros_like(numerator)
        correlations[valid_samples] = numerator[valid_samples] / denominator[valid_samples]

        # Check for NaN values in correlations
        if torch.isnan(correlations).any():
            self._log("Warning: NaN values detected in correlations", level="warning")
            # Replace NaNs with zeros for stability
            correlations = torch.nan_to_num(correlations, nan=0.0)

        # Calculate mean of valid correlations
        if torch.sum(valid_samples) > 0:
            avg_correlation = correlations[valid_samples].mean()
        else:
            avg_correlation = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Final NaN check on average correlation
        if torch.isnan(avg_correlation):
            self._log("Warning: NaN average correlation detected", level="warning")
            avg_correlation = torch.tensor(0.0, device=self.device, requires_grad=True)

        return avg_correlation

    def correlation_loss(self, original_ips_batch, masked_ips_batch):
        """
        Compute correlation loss for a batch.

        Args:
            original_ips_batch: Batch of original inner products
            masked_ips_batch: Batch of masked inner products

        Returns:
            Negative correlation (for minimization)
        """
        avg_correlation = self.compute_correlations(original_ips_batch, masked_ips_batch)

        # Return negative correlation (for minimization)
        return -avg_correlation

    def sparsity_loss(self):
        """
        Compute sparsity loss (L-1 loss) for the mask with adaptive regularization.

        Returns:
            Sparsity loss
        """
        mask_grad = self.sigmoid_grad()

        active_grad = torch.sum(mask_grad > 0.5).item()
        # Compute the L-1 norm of the mask
        L1_norm = torch.sum(mask_grad)

        # # Adaptive regularization factor
        if active_grad <= self.min_active_gradient * 2:
            grad_factor = max(0.1, active_grad / (self.min_active_gradient * 2))
        else:
            grad_factor = 1.0

        # Sparsity loss
        return self.lambda_reg * grad_factor * L1_norm

    def update_temperature(self, epoch, num_epochs):
        """
        Update temperature based on current epoch.

        Args:
            epoch: Current epoch
            num_epochs: Total number of epochs
        """
        # Exponential decay
        self.temperature = max(
            self.min_temperature,
            self.initial_temperature * math.exp(-self.temp_decay_rate * epoch)
        )
        return self.temperature

    def train_step_batch(self, test_grad_batch, train_grad_batch, original_ips_fn=None, accumulate_grad=False):
        """
        Perform one optimization step with batched data.

        Args:
            test_grad_batch: Batch of test gradients
            train_grad_batch: Batch of training gradients
            original_ips_fn: Function to compute original inner products if needed
            accumulate_grad: Whether to accumulate gradients without optimizer step

        Returns:
            Dictionary of metrics
        """
        if not accumulate_grad:
            self.optimizer.zero_grad()

        # Compute original inner products (without mask) if function provided
        if original_ips_fn is not None:
            original_ips_batch = original_ips_fn(test_grad_batch, train_grad_batch)
        else:
            original_ips_batch = self.compute_batch_inner_products(
                test_grad_batch, train_grad_batch, apply_mask=False
            )

        # Compute masked inner products
        masked_ips_batch = self.compute_batch_inner_products(
            test_grad_batch, train_grad_batch, apply_mask=True
        )

        # Compute correlation loss
        corr_loss = self.correlation_loss(original_ips_batch, masked_ips_batch)

        # Compute sparsity loss
        sparse_loss = self.sparsity_loss()

        # Total loss
        total_loss = corr_loss + sparse_loss

        # Compute gradients
        if total_loss.requires_grad:
            total_loss.backward()

            # Check for NaN gradients
            if torch.isnan(self.S_grad.grad).any():
                self._log("NaN gradients detected", level="warning")
                # Zero out NaN gradients
                self.S_grad.grad[torch.isnan(self.S_grad.grad)] = 0.0

        # Update parameters if not accumulating gradients
        if not accumulate_grad:
            self.optimizer.step()
            # Enforce minimum active parameters after optimizer step
            with torch.no_grad():
                self._enforce_minimum_active_params()

        # Compute sparsity statistics
        mask_grad = self.sigmoid_grad()
        grad_sparsity = (mask_grad < 0.5).float().mean().item()

        return {
            'total_loss': total_loss.item(),
            'correlation_loss': corr_loss.item(),
            'sparsity_loss': sparse_loss.item(),
            'gradient_sparsity': grad_sparsity,
            'batch_size': test_grad_batch.size(0)
        }

    def _create_dataloader(self, data, batch_size, shuffle=True):
        """Create a DataLoader from input data"""
        if isinstance(data, torch.Tensor):
            dataset = TensorDataset(data)
        else:
            data_tensor = self._ensure_tensor(data)
            dataset = TensorDataset(data_tensor)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train(
            self,
            train_gradients,
            test_gradients,
            batch_size=32,
            accumulation_steps=1,
            num_epochs=500,
            log_every=50,
            correlation_threshold=0.9
        ):
        """
        Train the gradient mask optimizer for multiple epochs with mini-batches.

        Args:
            train_gradients: Training gradients
            test_gradients: Test gradients
            batch_size: Size of mini-batches
            accumulation_steps: Number of steps to accumulate gradients
            num_epochs: Maximum number of training epochs
            log_every: Log progress every N epochs
            correlation_threshold: Stop training when correlation drops below this value

        Returns:
            Evaluation metrics
        """
        # Convert inputs to tensors if necessary and move to device
        train_gradients = self._ensure_tensor(train_gradients)
        test_gradients = self._ensure_tensor(test_gradients)

        # Create data loaders
        train_loader = self._create_dataloader(
            train_gradients, batch_size=min(batch_size, len(train_gradients))
        )

        test_loader = self._create_dataloader(
            test_gradients, batch_size=min(batch_size, len(test_gradients))
        )

        # Track the best mask during training (for logging purposes only)
        best_correlation = -float('inf')
        best_mask_info = None

        # Track masks that meet sparsity constraints (for logging purposes only)
        candidate_masks = []

        self._log(f"Starting training for {num_epochs} epochs with batch size {batch_size}")
        self._log(f"Gradient dimension: {self.gradient_dim}")
        self._log(f"Initial temperature: {self.temperature:.4f}")

        for epoch in range(num_epochs):
            # Update temperature for this epoch
            current_temp = self.update_temperature(epoch, num_epochs)

            total_loss = 0.0
            total_corr_loss = 0.0
            total_sparse_loss = 0.0
            total_batches = 0

            # Training loop with mini-batches
            for step, (train_grad_batch,) in enumerate(train_loader):
                # Use smaller batch from test set for each training step
                for test_step, (test_grad_batch,) in enumerate(test_loader):
                    # Break after processing one test batch to avoid excessive computation
                    if test_step > 0:
                        break

                    # Set gradient accumulation flag
                    accumulate_grad = (step % accumulation_steps != 0)

                    # Perform training step with current batch
                    metrics = self.train_step_batch(
                        test_grad_batch, train_grad_batch,
                        accumulate_grad=accumulate_grad
                    )

                    # Update running statistics
                    total_loss += metrics['total_loss']
                    total_corr_loss += metrics['correlation_loss']
                    total_sparse_loss += metrics['sparsity_loss']
                    total_batches += 1

                    # Update parameters if accumulation complete
                    if (step + 1) % accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # Enforce minimum active parameters
                        with torch.no_grad():
                            self._enforce_minimum_active_params()

            # Compute average metrics
            avg_loss = total_loss / max(1, total_batches)
            avg_corr_loss = total_corr_loss / max(1, total_batches)
            avg_sparse_loss = total_sparse_loss / max(1, total_batches)

            # Get current mask and count active parameters
            with torch.no_grad():
                mask_grad = self.sigmoid_grad()
                active_grad = torch.sum(mask_grad > 0.5).item()

            # Log progress periodically
            if epoch % log_every == 0 or epoch == num_epochs - 1:
                # Evaluate current mask performance
                eval_metrics = self.evaluate_mask_batched(
                    test_gradients, train_gradients,
                    batch_size=min(batch_size, len(test_gradients)),
                    verbose=False  # Don't log individual correlations
                )

                # Log progress
                self._log(f"Epoch {epoch}/{num_epochs} - "
                        f"Loss: {avg_loss:.4f}, "
                        f"Corr: {-avg_corr_loss:.4f}, "
                        f"Rank Corr: {eval_metrics['avg_rank_correlation']:.4f}, "
                        f"Grad: {active_grad}/{self.gradient_dim}, "
                        f"Temp: {current_temp:.4f}")

                # Check if mask satisfies constraints and correlation is better (for logging only)
                correlation_value = -avg_corr_loss
                meets_constraints = (
                    self.min_active_gradient <= active_grad <= self.max_active_gradient
                )

                if meets_constraints:
                    candidate_masks.append({
                        'correlation': correlation_value,
                        'active_grad': active_grad,
                        'mask_grad': self.S_grad.data.clone(),
                        'epoch': epoch,
                        'temperature': current_temp,
                        'avg_rank_correlation': eval_metrics.get('avg_rank_correlation', float('nan'))
                    })

                # Track best mask for logging purposes only
                if meets_constraints and correlation_value > best_correlation:
                    best_correlation = correlation_value
                    best_mask_info = {
                        'correlation': correlation_value,
                        'epoch': epoch,
                        'temperature': current_temp
                    }
                    self._log(f"New best mask at epoch {epoch} (correlation: {correlation_value:.4f}, temp: {current_temp:.4f})")

                # Early stopping based on correlation threshold
                avg_rank_correlation = eval_metrics.get('avg_rank_correlation', float('nan'))
                if not math.isnan(correlation_value) and avg_rank_correlation < correlation_threshold:
                    self._log(f"Early stopping at epoch {epoch} - correlation {avg_rank_correlation:.4f} below threshold {correlation_threshold:.4f}")
                    break

        # MODIFIED: Always use the final mask (don't replace it with the best mask)
        self._log("Using final mask from training (not the best mask)")

        # Log information about the best mask that was found (for comparison)
        if best_mask_info is not None:
            self._log(f"For reference, best mask was from epoch {best_mask_info['epoch']} "
                    f"(correlation: {best_mask_info['correlation']:.4f}, temp: {best_mask_info['temperature']:.4f})")

        # Set temperature to minimum for final evaluation to get closest to binary mask
        self.temperature = self.min_temperature
        self._log(f"Setting final temperature to {self.temperature:.4f} for evaluation")

        # Final evaluation
        self._log("Final mask evaluation:")
        eval_metrics = self.evaluate_mask_batched(
            test_gradients, train_gradients,
            batch_size=min(batch_size, len(test_gradients)),
            verbose=False
        )

        # Report if we met the correlation threshold
        if 'avg_rank_correlation' in eval_metrics and not math.isnan(eval_metrics['avg_rank_correlation']):
            if eval_metrics['avg_rank_correlation'] >= correlation_threshold:
                self._log(f"✓ Final rank correlation: {eval_metrics['avg_rank_correlation']:.4f} (above threshold {correlation_threshold:.4f})")
            else:
                self._log(f"✗ Final rank correlation: {eval_metrics['avg_rank_correlation']:.4f} (below threshold {correlation_threshold:.4f})")

        return eval_metrics

    def _ensure_tensor(self, x):
        """Convert input to tensor and move to device if needed"""
        if isinstance(x, list):
            if isinstance(x[0], torch.Tensor):
                return torch.stack(x).to(self.device)
            else:
                return torch.tensor(x).to(self.device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")

    def _enforce_minimum_active_params(self):
        """Enforce minimum active parameters for the mask"""
        mask_grad = self.sigmoid_grad()
        active_grad = torch.sum(mask_grad > 0.5).item()

        # Force minimum gradient parameters to be active if needed
        if active_grad < self.min_active_gradient:
            top_k_values, top_k_indices = torch.topk(mask_grad, k=self.min_active_gradient)
            new_S_grad = self.S_grad.data.clone()
            boost_amount = 2.0 * self.temperature  # Scale boost by temperature
            new_S_grad[top_k_indices] = boost_amount
            self.S_grad.data = new_S_grad
            self._log(f"Forced {self.min_active_gradient} gradient parameters to be active", level="warning")

    def evaluate_mask_batched(self, test_gradients, train_gradients, batch_size=32, verbose=False):
        """
        Evaluate the current mask's performance in preserving rankings, processing data in batches.

        Args:
            test_gradients: Test gradients
            train_gradients: Training gradients
            batch_size: Size of mini-batches
            verbose: Whether to log detailed per-sample correlations

        Returns:
            Dictionary of evaluation metrics
        """
        with torch.no_grad():
            # Get mask stats
            mask_grad = self.sigmoid_grad()
            active_grad = torch.sum(mask_grad > 0.5).int().item()
            percent_active_grad = active_grad / mask_grad.numel() * 100

            if verbose:
                self._log(f"Gradient Mask: {active_grad}/{mask_grad.numel()} parameters active ({percent_active_grad:.2f}%)")
                self._log(f"Current temperature: {self.temperature:.4f}")

            # Process test data in batches
            all_correlations = []

            for test_grad_batch, in self._create_dataloader(test_gradients, batch_size, shuffle=False):
                # Compute inner products for this test batch against all training data
                original_ips_batch = []
                masked_ips_batch = []

                for train_grad_batch, in self._create_dataloader(train_gradients, batch_size, shuffle=False):
                    # Compute batch inner products
                    orig_ips = self.compute_batch_inner_products(
                        test_grad_batch, train_grad_batch, apply_mask=False
                    )

                    mask_ips = self.compute_batch_inner_products(
                        test_grad_batch, train_grad_batch, apply_mask=True
                    )

                    # Append to batch results
                    original_ips_batch.append(orig_ips)
                    masked_ips_batch.append(mask_ips)

                # Concatenate inner products from all training batches
                if original_ips_batch:
                    # Handle different batch sizes by padding to max width
                    max_width = max(ips.size(1) for ips in original_ips_batch)

                    # Concatenate batches along train dimension (column-wise)
                    original_ips_concat = torch.cat([
                        torch.nn.functional.pad(ips, (0, max_width - ips.size(1)))
                        for ips in original_ips_batch
                    ], dim=1)

                    masked_ips_concat = torch.cat([
                        torch.nn.functional.pad(ips, (0, max_width - ips.size(1)))
                        for ips in masked_ips_batch
                    ], dim=1)

                    # Convert to CPU for Spearman calculation
                    original_ips_np = original_ips_concat.cpu().numpy()
                    masked_ips_np = masked_ips_concat.cpu().numpy()

                    # Compute Spearman rank correlation for each test sample in batch
                    for i in range(original_ips_np.shape[0]):
                        # Remove padded zeros if any
                        valid_cols = (original_ips_np[i] != 0).sum() if max_width > train_gradients.size(0) else original_ips_np.shape[1]

                        orig_row = original_ips_np[i, :valid_cols]
                        masked_row = masked_ips_np[i, :valid_cols]

                        try:
                            rank_corr, _ = spearmanr(orig_row, masked_row)
                            all_correlations.append(rank_corr)
                            if verbose:
                                self._log(f"Test Sample {len(all_correlations)}: Spearman Rank Correlation: {rank_corr:.4f}")
                        except:
                            if verbose:
                                self._log(f"Test Sample {len(all_correlations) + 1}: Could not compute correlation.", level="warning")

            # Compute average correlation
            avg_correlation = float('nan')
            if all_correlations:
                avg_correlation = sum(all_correlations) / len(all_correlations)

            return {
                'rank_correlations': all_correlations,
                'avg_rank_correlation': avg_correlation,
                'active_grad': active_grad,
                'percent_active_grad': percent_active_grad,
                'current_temperature': self.temperature
            }

    def get_mask_hardness(self):
        """
        Calculate how close the mask is to being binary (0/1).

        Returns:
            Float between 0 and 1, where 1 means completely binary
        """
        mask_grad = self.sigmoid_grad()
        # Calculate how far values are from 0.5 (middle point)
        distances = torch.abs(mask_grad - 0.5) * 2  # Scale to [0,1]
        # Average distance (1.0 means all values are exactly 0 or 1)
        return distances.mean().item()

    def get_important_indices(self, threshold=0.5, min_count=None):
        """
        Get indices of important parameters for the gradient mask.

        Args:
            threshold: Value threshold for selecting parameters
            min_count: Minimum number of gradient parameters to select

        Returns:
            Dictionary with important indices for the mask
        """
        mask_grad = self.sigmoid_grad()

        # Get mask hardness metrics
        mask_hardness = self.get_mask_hardness()
        self._log(f"Mask hardness: {mask_hardness:.4f} (1.0 is completely binary)")

        # Get indices for gradient mask
        grad_indices = torch.where(mask_grad > threshold)[0]
        if min_count is not None and len(grad_indices) < min_count:
            self._log(f"Warning: Only {len(grad_indices)} gradient parameters above threshold. Selecting top-{min_count} instead.", level="warning")
            values, top_indices = torch.topk(mask_grad, k=min_count)
            grad_indices = top_indices
        if len(grad_indices) > self.max_active_gradient:
            self._log(f"Warning: More than {self.max_active_gradient} gradient parameters selected. Reducing to max.", level="warning")
            values, top_indices = torch.topk(mask_grad, k=self.max_active_gradient)
            grad_indices = top_indices

        # Log summary of selected indices
        total_params = self.gradient_dim
        sparsity = 100 - (len(grad_indices) / total_params * 100)

        self._log(f"Selected {len(grad_indices)} gradient indices")
        self._log(f"Effective parameters: {len(grad_indices)}/{total_params} ({100-sparsity:.2f}% of total)")
        self._log(f"Sparsity achieved: {sparsity:.2f}%")
        self._log(f"Current temperature: {self.temperature:.4f}")

        return {
            'gradient': grad_indices,
            'mask_hardness': mask_hardness
        }