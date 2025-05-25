import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr
import math
from torch.utils.data import DataLoader, TensorDataset

class MLPSelectiveMask:
    def __init__(
            self,
            pre_activation_dim,
            input_features_dim,
            lambda_reg=1e-1,
            lr=0.01,
            min_active_pre_activation=10,
            max_active_pre_activation=None,
            min_active_input=10,
            max_active_input=None,
            device='cpu',
            logger=None
        ):
        """
        Initialize the dual component mask optimizer.

        Args:
            pre_activation_dim: Dimensionality of the pre-activation gradients
            input_features_dim: Dimensionality of the input features
            lambda_reg: Regularization parameter for sparsity
            lr: Learning rate for optimizer
            min_active_pre_activation: Minimum number of pre-activation components that must remain active
            max_active_pre_activation: Maximum pre-activation components allowed
            min_active_input: Minimum number of input feature components that must remain active
            max_active_input: Maximum input feature components allowed
            device: Device to run computations on
            logger: Logger object for output messages
        """
        self.device = device
        self.lambda_reg = lambda_reg
        self.logger = logger

        # Dimensions
        self.pre_activation_dim = pre_activation_dim
        self.input_features_dim = input_features_dim

        # Constraints for each mask
        self.min_active_pre_activation = min_active_pre_activation
        self.max_active_pre_activation = max_active_pre_activation if max_active_pre_activation is not None else 2 * min_active_pre_activation

        self.min_active_input = min_active_input
        self.max_active_input = max_active_input if max_active_input is not None else 2 * min_active_input

        # Initialize mask parameters for pre-activation and input features
        self.S_pre = nn.Parameter(torch.randn(pre_activation_dim, device=device) * 0.01)
        self.S_input = nn.Parameter(torch.randn(input_features_dim, device=device) * 0.01)

        # Use Adam optimizer for both masks
        self.optimizer = optim.Adam([self.S_pre, self.S_input], lr=lr)

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

    def sigmoid_pre(self):
        """Return sigmoid(S_pre) as the pre-activation mask"""
        return torch.sigmoid(self.S_pre)

    def sigmoid_input(self):
        """Return sigmoid(S_input) as the input features mask"""
        return torch.sigmoid(self.S_input)

    def construct_gradient(self, pre_activation, input_features, apply_mask=False, is_3d=False):
        """
        Construct gradients from components, optionally applying masks.

        Args:
            pre_activation: Pre-activation gradients [batch_size, pre_activation_dim] or [batch_size, seq_len, pre_activation_dim]
            input_features: Input features [batch_size, input_features_dim] or [batch_size, seq_len, input_features_dim]
            apply_mask: Whether to apply masks to components
            is_3d: Whether inputs are 3D tensors (for handling sequence data)

        Returns:
            Constructed gradients [batch_size, (pre_activation_dim * input_features_dim)]
        """
        if apply_mask:
            mask_pre = self.sigmoid_pre()
            mask_input = self.sigmoid_input()

            if is_3d:
                # For 3D tensors, expand masks to match sequence dimension
                mask_pre = mask_pre.unsqueeze(0)  # [1, pre_dim]
                mask_input = mask_input.unsqueeze(0)  # [1, input_dim]

                # Apply masks to the last dimension
                pre_activation = pre_activation * mask_pre
                input_features = input_features * mask_input
            else:
                # For 2D tensors, simple element-wise multiplication
                pre_activation = pre_activation * mask_pre
                input_features = input_features * mask_input

        # Construct gradients using einsum
        batch_size = pre_activation.shape[0]
        if is_3d:
            grad = torch.einsum('ijk,ijl->ikl', pre_activation, input_features).reshape(batch_size, -1)
        else:
            grad = torch.einsum('bi,bj->bij', pre_activation, input_features).reshape(batch_size, -1)

        return grad

    def compute_batch_inner_products(self, test_pre_batch, test_input_batch, train_pre_batch, train_input_batch, apply_mask=False, is_3d=False):
        """
        Compute inner products between constructed test and training gradients for a single batch.

        Args:
            test_pre_batch: Batch of test pre-activation gradients
            test_input_batch: Batch of test input features
            train_pre_batch: Batch of training pre-activation gradients
            train_input_batch: Batch of training input features
            apply_mask: Whether to apply masks to components
            is_3d: Whether inputs are 3D tensors

        Returns:
            Tensor of inner products [batch_size_test, batch_size_train]
        """
        # Construct test gradients
        test_grads = self.construct_gradient(test_pre_batch, test_input_batch, apply_mask, is_3d)

        # Construct training gradients
        train_grads = self.construct_gradient(train_pre_batch, train_input_batch, apply_mask, is_3d)

        # Compute inner products using matrix multiplication
        return torch.matmul(test_grads, train_grads.T)

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
        epsilon = 1e-5

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
        Compute sparsity loss for both masks with adaptive regularization.

        Returns:
            Combined sparsity loss
        """
        mask_pre = self.sigmoid_pre()
        mask_input = self.sigmoid_input()

        # Count active parameters in each mask
        active_pre = torch.sum(mask_pre > 0.5).item()
        active_input = torch.sum(mask_input > 0.5).item()

        # Adaptive regularization factors
        if active_pre <= self.min_active_pre_activation * 2:
            pre_factor = max(0.1, active_pre / (self.min_active_pre_activation * 2))
        else:
            pre_factor = 1.0

        if active_input <= self.min_active_input * 2:
            input_factor = max(0.1, active_input / (self.min_active_input * 2))
        else:
            input_factor = 1.0

        # Combined sparsity loss
        pre_loss = self.lambda_reg * pre_factor * torch.sum(mask_pre)
        input_loss = self.lambda_reg * input_factor * torch.sum(mask_input)

        return pre_loss + input_loss

    def train_step_batch(self, test_pre_batch, test_input_batch, train_pre_batch, train_input_batch,
                          original_ips_fn=None, is_3d=False, accumulate_grad=False):
        """
        Perform one optimization step with batched data.

        Args:
            test_pre_batch: Batch of test pre-activation gradients
            test_input_batch: Batch of test input features
            train_pre_batch: Batch of training pre-activation gradients
            train_input_batch: Batch of training input features
            original_ips_fn: Function to compute original inner products if needed
            is_3d: Whether inputs are 3D tensors
            accumulate_grad: Whether to accumulate gradients without optimizer step

        Returns:
            Dictionary of metrics
        """
        if not accumulate_grad:
            self.optimizer.zero_grad()

        # Compute original inner products (without masks) if function provided
        if original_ips_fn is not None:
            original_ips_batch = original_ips_fn(test_pre_batch, test_input_batch, train_pre_batch, train_input_batch)
        else:
            original_ips_batch = self.compute_batch_inner_products(
                test_pre_batch, test_input_batch, train_pre_batch, train_input_batch,
                apply_mask=False, is_3d=is_3d
            )

        # Compute masked inner products
        masked_ips_batch = self.compute_batch_inner_products(
            test_pre_batch, test_input_batch, train_pre_batch, train_input_batch,
            apply_mask=True, is_3d=is_3d
        )

        # Compute correlation loss
        corr_loss = self.correlation_loss(original_ips_batch, masked_ips_batch)

        # Compute sparsity loss
        sparse_loss = self.sparsity_loss()

        # Total loss
        total_loss = corr_loss + sparse_loss

        # # Compute gradients
        # total_loss.backward()

        # Add gradient norm logging and clipping
        if total_loss.requires_grad:
            total_loss.backward()

            # Check for NaN gradients
            if torch.isnan(self.S_pre.grad).any() or torch.isnan(self.S_input.grad).any():
                self._log("NaN gradients detected", level="warning")
                # Zero out NaN gradients
                self.S_pre.grad[torch.isnan(self.S_pre.grad)] = 0.0
                self.S_input.grad[torch.isnan(self.S_input.grad)] = 0.0

        # Update parameters if not accumulating gradients
        if not accumulate_grad:
            self.optimizer.step()
            # Enforce minimum active parameters after optimizer step
            with torch.no_grad():
                self._enforce_minimum_active_params()

        # Compute sparsity statistics
        mask_pre = self.sigmoid_pre()
        mask_input = self.sigmoid_input()
        pre_sparsity = (mask_pre < 0.5).float().mean().item()
        input_sparsity = (mask_input < 0.5).float().mean().item()

        return {
            'total_loss': total_loss.item(),
            'correlation_loss': corr_loss.item(),
            'sparsity_loss': sparse_loss.item(),
            'pre_activation_sparsity': pre_sparsity,
            'input_features_sparsity': input_sparsity,
            'batch_size': test_pre_batch.size(0)
        }

    def _create_dataloader(self, data, batch_size, shuffle=True):
        """Create a DataLoader from input data"""
        if isinstance(data, torch.Tensor):
            dataset = TensorDataset(data)
        else:
            data_tensor = self._ensure_tensor(data)
            dataset = TensorDataset(data_tensor)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _create_paired_dataloader(self, data1, data2, batch_size, shuffle=True):
        """Create a DataLoader from paired input data"""
        data1_tensor = self._ensure_tensor(data1)
        data2_tensor = self._ensure_tensor(data2)
        dataset = TensorDataset(data1_tensor, data2_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train(
            self,
            train_pre_activation,
            train_input_features,
            test_pre_activation,
            test_input_features,
            batch_size=32,
            accumulation_steps=1,
            num_epochs=500,
            log_every=50,
            correlation_threshold=0.9,
            is_3d=False
        ):
        """
        Train the dual mask optimizer for multiple epochs with mini-batches.

        Args:
            train_pre_activation: Training pre-activation gradients
            train_input_features: Training input features
            test_pre_activation: Test pre-activation gradients
            test_input_features: Test input features
            batch_size: Size of mini-batches
            accumulation_steps: Number of steps to accumulate gradients
            num_epochs: Maximum number of training epochs
            log_every: Log progress every N epochs
            correlation_threshold: Stop training when correlation drops below this value
            is_3d: Whether inputs are 3D tensors (for sequence data)

        Returns:
            Evaluation metrics
        """
        # Convert inputs to tensors if necessary and move to device
        train_pre_activation = self._ensure_tensor(train_pre_activation)
        train_input_features = self._ensure_tensor(train_input_features)
        test_pre_activation = self._ensure_tensor(test_pre_activation)
        test_input_features = self._ensure_tensor(test_input_features)

        # Create data loaders
        train_loader = self._create_paired_dataloader(
            train_pre_activation, train_input_features,
            batch_size=min(batch_size, len(train_pre_activation))
        )

        test_loader = self._create_paired_dataloader(
            test_pre_activation, test_input_features,
            batch_size=min(batch_size, len(test_pre_activation))
        )

        # For tracking purposes only - we will use the final mask regardless
        best_correlation = -float('inf')
        best_masks = None

        # Track masks that meet sparsity constraints
        candidate_masks = []

        self._log(f"Starting training for {num_epochs} epochs with batch size {batch_size}")
        self._log(f"Pre-activation dimension: {self.pre_activation_dim}, Input features dimension: {self.input_features_dim}")

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_corr_loss = 0.0
            total_sparse_loss = 0.0
            total_batches = 0

            # Training loop with mini-batches
            for step, ((train_pre_batch,), (train_input_batch,)) in enumerate(zip(
                self._create_dataloader(train_pre_activation, batch_size),
                self._create_dataloader(train_input_features, batch_size)
            )):
                # Use smaller batch from test set for each training step
                for test_step, ((test_pre_batch,), (test_input_batch,)) in enumerate(zip(
                    self._create_dataloader(test_pre_activation, min(batch_size, len(test_pre_activation))),
                    self._create_dataloader(test_input_features, min(batch_size, len(test_input_features)))
                )):
                    # Break after processing one test batch to avoid excessive computation
                    if test_step > 0:
                        break

                    # Set gradient accumulation flag
                    accumulate_grad = (step % accumulation_steps != 0)

                    # Perform training step with current batch
                    metrics = self.train_step_batch(
                        test_pre_batch, test_input_batch,
                        train_pre_batch, train_input_batch,
                        is_3d=is_3d,
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

            # Get current masks and count active parameters
            with torch.no_grad():
                mask_pre = self.sigmoid_pre()
                mask_input = self.sigmoid_input()
                active_pre = torch.sum(mask_pre > 0.5).item()
                active_input = torch.sum(mask_input > 0.5).item()

            # Log progress periodically
            if epoch % log_every == 0 or epoch == num_epochs - 1:
                # Evaluate current mask performance
                eval_metrics = self.evaluate_masks_batched(
                    test_pre_activation, test_input_features,
                    train_pre_activation, train_input_features,
                    batch_size=min(batch_size, len(test_pre_activation)),
                    is_3d=is_3d,
                    verbose=False  # Don't log individual correlations
                )

                # Log progress
                self._log(f"Epoch {epoch}/{num_epochs} - "
                         f"Loss: {avg_loss:.4f}, "
                         f"Corr: {-avg_corr_loss:.4f}, "
                         f"Rank Corr: {eval_metrics['avg_rank_correlation']:.4f}, "
                         f"Pre: {active_pre}/{self.pre_activation_dim}, "
                         f"Input: {active_input}/{self.input_features_dim}")

                # Check if masks satisfy constraints and correlation is better
                correlation_value = -avg_corr_loss
                meets_constraints = (
                    self.min_active_pre_activation <= active_pre <= self.max_active_pre_activation and
                    self.min_active_input <= active_input <= self.max_active_input
                )

                if meets_constraints:
                    candidate_masks.append({
                        'correlation': correlation_value,
                        'active_pre': active_pre,
                        'active_input': active_input,
                        'mask_pre': self.S_pre.data.clone(),
                        'mask_input': self.S_input.data.clone(),
                        'epoch': epoch,
                        'avg_rank_correlation': eval_metrics.get('avg_rank_correlation', float('nan'))
                    })

                # Track best mask for logging/reference purposes only
                if meets_constraints and correlation_value > best_correlation:
                    best_correlation = correlation_value
                    best_masks = {
                        'pre': self.S_pre.data.clone(),
                        'input': self.S_input.data.clone()
                    }
                    self._log(f"New best masks at epoch {epoch} (correlation: {correlation_value:.4f})")

                # Early stopping based on correlation threshold
                avg_rank_correlation = eval_metrics.get('avg_rank_correlation', float('nan'))
                if not math.isnan(correlation_value) and avg_rank_correlation < correlation_threshold:
                    self._log(f"Early stopping at epoch {epoch} - correlation {avg_rank_correlation:.4f} below threshold {correlation_threshold:.4f}")
                    break

        # MODIFIED: Always use the final mask (don't replace it with the best mask)
        self._log("Using final masks from training (not the best masks)")

        # Log information about the best mask that was found (for comparison)
        if best_masks is not None:
            self._log(f"For reference, best masks were found during training "
                    f"(correlation: {best_correlation:.4f})")

        # Final evaluation
        self._log("Final mask evaluation:")
        eval_metrics = self.evaluate_masks_batched(
            test_pre_activation, test_input_features,
            train_pre_activation, train_input_features,
            batch_size=min(batch_size, len(test_pre_activation)),
            is_3d=is_3d,
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
        """Enforce minimum active parameters for both masks"""
        mask_pre = self.sigmoid_pre()
        mask_input = self.sigmoid_input()

        active_pre = torch.sum(mask_pre > 0.5).item()
        active_input = torch.sum(mask_input > 0.5).item()

        # Force minimum pre-activation parameters to be active if needed
        if active_pre < self.min_active_pre_activation:
            top_k_values, top_k_indices = torch.topk(mask_pre, k=self.min_active_pre_activation)
            new_S_pre = self.S_pre.data.clone()
            boost_amount = 5.0
            new_S_pre[top_k_indices] = boost_amount
            self.S_pre.data = new_S_pre
            self._log(f"Forced {self.min_active_pre_activation} pre-activation parameters to be active", level="warning")

        # Force minimum input feature parameters to be active if needed
        if active_input < self.min_active_input:
            top_k_values, top_k_indices = torch.topk(mask_input, k=self.min_active_input)
            new_S_input = self.S_input.data.clone()
            boost_amount = 5.0
            new_S_input[top_k_indices] = boost_amount
            self.S_input.data = new_S_input
            self._log(f"Forced {self.min_active_input} input feature parameters to be active", level="warning")

    def evaluate_masks_batched(self, test_pre, test_input, train_pre, train_input, batch_size=32, is_3d=False, verbose=False):
        """
        Evaluate the current masks' performance in preserving rankings, processing data in batches.

        Args:
            test_pre: Test pre-activation gradients
            test_input: Test input features
            train_pre: Training pre-activation gradients
            train_input: Training input features
            batch_size: Size of mini-batches
            is_3d: Whether inputs are 3D tensors
            verbose: Whether to log detailed per-sample correlations

        Returns:
            Dictionary of evaluation metrics
        """
        with torch.no_grad():
            # Get mask stats
            mask_pre = self.sigmoid_pre()
            mask_input = self.sigmoid_input()

            active_pre = torch.sum(mask_pre > 0.5).int().item()
            active_input = torch.sum(mask_input > 0.5).int().item()

            percent_active_pre = active_pre / mask_pre.numel() * 100
            percent_active_input = active_input / mask_input.numel() * 100

            # Total active parameters in the full gradient
            total_active = active_pre * active_input
            total_possible = mask_pre.numel() * mask_input.numel()
            percent_active_total = total_active / total_possible * 100

            if verbose:
                self._log(f"Pre-activation Mask: {active_pre}/{mask_pre.numel()} parameters active ({percent_active_pre:.2f}%)")
                self._log(f"Input Features Mask: {active_input}/{mask_input.numel()} parameters active ({percent_active_input:.2f}%)")
                self._log(f"Total effective parameters: {total_active}/{total_possible} ({100-percent_active_total:.2f}%)")

            # Process test data in batches
            all_correlations = []

            for (test_pre_batch,), (test_input_batch,) in zip(
                self._create_dataloader(test_pre, batch_size, shuffle=False),
                self._create_dataloader(test_input, batch_size, shuffle=False)
            ):
                # Compute inner products for this test batch against all training data
                original_ips_batch = []
                masked_ips_batch = []

                for (train_pre_batch,), (train_input_batch,) in zip(
                    self._create_dataloader(train_pre, batch_size, shuffle=False),
                    self._create_dataloader(train_input, batch_size, shuffle=False)
                ):
                    # Compute batch inner products
                    orig_ips = self.compute_batch_inner_products(
                        test_pre_batch, test_input_batch,
                        train_pre_batch, train_input_batch,
                        apply_mask=False, is_3d=is_3d
                    )

                    mask_ips = self.compute_batch_inner_products(
                        test_pre_batch, test_input_batch,
                        train_pre_batch, train_input_batch,
                        apply_mask=True, is_3d=is_3d
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
                        valid_cols = (original_ips_np[i] != 0).sum() if max_width > train_pre.size(0) else original_ips_np.shape[1]

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
                'active_pre': active_pre,
                'active_input': active_input,
                'percent_active_pre': percent_active_pre,
                'percent_active_input': percent_active_input,
                'total_active': total_active,
                'percent_active_total': percent_active_total
            }

    def get_important_indices(self, threshold=0.5, min_count_pre=None, min_count_input=None):
        """
        Get indices of important parameters for both masks.

        Args:
            threshold: Value threshold for selecting parameters
            min_count_pre: Minimum number of pre-activation parameters to select
            min_count_input: Minimum number of input feature parameters to select

        Returns:
            Dictionary with important indices for both masks
        """
        mask_pre = self.sigmoid_pre()
        mask_input = self.sigmoid_input()

        # Get indices for pre-activation mask
        pre_indices = torch.where(mask_pre > threshold)[0]
        if min_count_pre is not None and len(pre_indices) < min_count_pre:
            self._log(f"Warning: Only {len(pre_indices)} pre-activation parameters above threshold. Selecting top-{min_count_pre} instead.", level="warning")
            values, top_indices = torch.topk(mask_pre, k=min_count_pre)
            pre_indices = top_indices
        if len(pre_indices) > self.max_active_pre_activation:
            self._log(f"Warning: More than {self.max_active_pre_activation} pre-activation parameters selected. Reducing to max.", level="warning")
            values, top_indices = torch.topk(mask_pre, k=self.max_active_pre_activation)
            pre_indices = top_indices

        # Get indices for input features mask
        input_indices = torch.where(mask_input > threshold)[0]
        if min_count_input is not None and len(input_indices) < min_count_input:
            self._log(f"Warning: Only {len(input_indices)} input feature parameters above threshold. Selecting top-{min_count_input} instead.", level="warning")
            values, top_indices = torch.topk(mask_input, k=min_count_input)
            input_indices = top_indices
        if len(input_indices) > self.max_active_input:
            self._log(f"Warning: More than {self.max_active_input} input feature parameters selected. Reducing to max.", level="warning")
            values, top_indices = torch.topk(mask_input, k=self.max_active_input)
            input_indices = top_indices

        # Log summary of selected indices
        effective_params = len(pre_indices) * len(input_indices)
        total_params = self.pre_activation_dim * self.input_features_dim
        sparsity = 100 - (effective_params / total_params * 100)

        self._log(f"Selected {len(pre_indices)} pre-activation indices and {len(input_indices)} input feature indices")
        self._log(f"Effective parameters: {effective_params}/{total_params} ({100-sparsity:.2f}% of total)")
        self._log(f"Sparsity achieved: {sparsity:.2f}%")

        return {
            'pre_activation': pre_indices,
            'input_features': input_indices
        }