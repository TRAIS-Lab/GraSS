import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr
import math

class DualComponentMaskOptimizer:
    def __init__(
            self,
            pre_activation_dim,
            input_features_dim,
            lambda_reg=0.001,
            lr=0.01,
            min_active_pre_activation=10,
            max_active_pre_activation=None,
            min_active_input=10,
            max_active_input=None,
            device='cuda' if torch.cuda.is_available() else 'cpu',
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

        # Cache for efficient computation
        self.original_ips_cache = None

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

    def compute_inner_products(self, test_pre, test_input, train_pre, train_input, apply_mask=False, is_3d=False):
        """
        Compute inner products between constructed test and training gradients.

        Args:
            test_pre: Test pre-activation gradients
            test_input: Test input features
            train_pre: Training pre-activation gradients
            train_input: Training input features
            apply_mask: Whether to apply masks to components
            is_3d: Whether inputs are 3D tensors

        Returns:
            Tensor of inner products [n_test, n_train]
        """
        # Construct test gradients
        test_grads = self.construct_gradient(test_pre, test_input, apply_mask, is_3d)

        # Construct training gradients
        train_grads = self.construct_gradient(train_pre, train_input, apply_mask, is_3d)

        # Compute inner products using matrix multiplication
        return torch.matmul(test_grads, train_grads.T)

    def correlation_loss(self, original_ips, masked_ips):
        """
        Compute average correlation between original and masked inner products.

        Args:
            original_ips: Tensor of original inner products [n_test, n_train]
            masked_ips: Tensor of masked inner products [n_test, n_train]

        Returns:
            Average negative correlation (for minimization)
        """
        # Mean center both sets of inner products along train dimension
        orig_centered = original_ips - original_ips.mean(dim=1, keepdim=True)
        masked_centered = masked_ips - masked_ips.mean(dim=1, keepdim=True)

        # Compute correlation for each test gradient
        numerator = torch.sum(orig_centered * masked_centered, dim=1)
        denominator = torch.sqrt(torch.sum(orig_centered**2, dim=1) * torch.sum(masked_centered**2, dim=1) + 1e-8)
        correlations = numerator / denominator

        # Average correlation across all test gradients
        avg_correlation = correlations.mean()

        # Return negative correlation (to minimize)
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

    def train_step(self, test_pre, test_input, train_pre, train_input, is_3d=False):
        """
        Perform one optimization step.

        Args:
            test_pre: Test pre-activation gradients
            test_input: Test input features
            train_pre: Training pre-activation gradients
            train_input: Training input features
            is_3d: Whether inputs are 3D tensors

        Returns:
            Dictionary of metrics
        """
        self.optimizer.zero_grad()

        # Compute original inner products (without masks)
        # Cache original inner products if not already computed
        if self.original_ips_cache is None:
            original_ips = self.compute_inner_products(
                test_pre, test_input, train_pre, train_input,
                apply_mask=False, is_3d=is_3d
            )
            self.original_ips_cache = original_ips
        else:
            original_ips = self.original_ips_cache

        # Compute masked inner products
        masked_ips = self.compute_inner_products(
            test_pre, test_input, train_pre, train_input,
            apply_mask=True, is_3d=is_3d
        )

        # Compute correlation loss
        corr_loss = self.correlation_loss(original_ips, masked_ips)

        # Compute sparsity loss
        sparse_loss = self.sparsity_loss()

        # Total loss
        total_loss = corr_loss + sparse_loss

        # Compute gradients and update parameters
        total_loss.backward()
        self.optimizer.step()

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
            'input_features_sparsity': input_sparsity
        }

    def train(self,
              train_pre_activation,
              train_input_features,
              test_pre_activation,
              test_input_features,
              num_epochs=500,
              log_every=50,
              correlation_threshold=0.7,
              is_3d=False):
        """
        Train the dual mask optimizer for multiple epochs.

        Args:
            train_pre_activation: Training pre-activation gradients
            train_input_features: Training input features
            test_pre_activation: Test pre-activation gradients
            test_input_features: Test input features
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

        # Reset caches
        self.original_ips_cache = None

        best_correlation = -float('inf')
        best_masks = None

        # Track masks that meet sparsity constraints
        candidate_masks = []

        self._log(f"Starting training for {num_epochs} epochs (early stopping threshold: {correlation_threshold})")
        self._log(f"Pre-activation dimension: {self.pre_activation_dim}, Input features dimension: {self.input_features_dim}")

        for epoch in range(num_epochs):
            metrics = self.train_step(
                test_pre_activation, test_input_features,
                train_pre_activation, train_input_features,
                is_3d=is_3d
            )

            # Safety check for minimum active parameters
            with torch.no_grad():
                self._enforce_minimum_active_params()

                # Get current masks and count active parameters
                mask_pre = self.sigmoid_pre()
                mask_input = self.sigmoid_input()
                active_pre = torch.sum(mask_pre > 0.5).item()
                active_input = torch.sum(mask_input > 0.5).item()

            # Log progress periodically
            if epoch % log_every == 0 or epoch == num_epochs - 1:
                # Evaluate current mask performance
                eval_metrics = self.evaluate_masks(
                    test_pre_activation, test_input_features,
                    train_pre_activation, train_input_features,
                    is_3d=is_3d,
                    verbose=False  # Don't log individual correlations
                )

                # Log progress
                self._log(f"Epoch {epoch}/{num_epochs} - "
                         f"Loss: {metrics['total_loss']:.4f}, "
                         f"Corr: {-metrics['correlation_loss']:.4f}, "
                         f"Rank Corr: {eval_metrics['avg_rank_correlation']:.4f}, "
                         f"Pre: {active_pre}/{self.pre_activation_dim}, "
                         f"Input: {active_input}/{self.input_features_dim}")

                # Check if masks satisfy constraints and correlation is better
                correlation_value = -metrics['correlation_loss']
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

                if meets_constraints and correlation_value > best_correlation:
                    best_correlation = correlation_value
                    best_masks = {
                        'pre': self.S_pre.data.clone(),
                        'input': self.S_input.data.clone()
                    }
                    self._log(f"New best masks at epoch {epoch} (correlation: {correlation_value:.4f})")

                # Early stopping based on correlation threshold
                avg_rank_correlation = eval_metrics.get('avg_rank_correlation', float('nan'))
                if not math.isnan(avg_rank_correlation) and avg_rank_correlation < correlation_threshold:
                    self._log(f"Early stopping at epoch {epoch} - correlation {avg_rank_correlation:.4f} below threshold {correlation_threshold:.4f}")
                    break

        # Select the best masks
        if best_masks is not None:
            self._log("Using best masks found during training")
            self.S_pre = nn.Parameter(best_masks['pre'])
            self.S_input = nn.Parameter(best_masks['input'])
        elif len(candidate_masks) > 0:
            best_candidate = max(candidate_masks, key=lambda x: x['correlation'])
            self._log(f"Using best candidate masks from epoch {best_candidate['epoch']} "
                    f"(correlation: {best_candidate['correlation']:.4f})")
            self.S_pre = nn.Parameter(best_candidate['mask_pre'])
            self.S_input = nn.Parameter(best_candidate['mask_input'])
        else:
            self._log("Warning: No masks met the sparsity constraints. Using final masks.")

        # Final evaluation
        self._log("Final mask evaluation:")
        eval_metrics = self.evaluate_masks(
            test_pre_activation, test_input_features,
            train_pre_activation, train_input_features,
            is_3d=is_3d,
            verbose=False
        )

        # Report if we met the correlation threshold
        if 'avg_rank_correlation' in eval_metrics and not math.isnan(eval_metrics['avg_rank_correlation']):
            if eval_metrics['avg_rank_correlation'] >= correlation_threshold:
                self._log(f"✓ Final correlation: {eval_metrics['avg_rank_correlation']:.4f} (above threshold {correlation_threshold:.4f})")
            else:
                self._log(f"✗ Final correlation: {eval_metrics['avg_rank_correlation']:.4f} (below threshold {correlation_threshold:.4f})")

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

    def evaluate_masks(self, test_pre, test_input, train_pre, train_input, is_3d=False, verbose=False):
        """
        Evaluate the current masks' performance in preserving rankings.

        Args:
            test_pre: Test pre-activation gradients
            test_input: Test input features
            train_pre: Training pre-activation gradients
            train_input: Training input features
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
                self._log(f"Total effective parameters: {total_active}/{total_possible} ({percent_active_total:.2f}%)")

            # Compute inner products
            original_ips = self.compute_inner_products(
                test_pre, test_input, train_pre, train_input,
                apply_mask=False, is_3d=is_3d
            )
            masked_ips = self.compute_inner_products(
                test_pre, test_input, train_pre, train_input,
                apply_mask=True, is_3d=is_3d
            )

            # Convert to CPU for Spearman calculation
            original_ips_np = original_ips.cpu().numpy()
            masked_ips_np = masked_ips.cpu().numpy()

            # Compute Spearman rank correlation for each test gradient
            all_correlations = []
            n_test = original_ips.shape[0]

            for i in range(n_test):
                try:
                    rank_corr, _ = spearmanr(original_ips_np[i], masked_ips_np[i])
                    all_correlations.append(rank_corr)
                    if verbose:
                        self._log(f"Test Sample {i+1}: Spearman Rank Correlation: {rank_corr:.4f}")
                except:
                    if verbose:
                        self._log(f"Test Sample {i+1}: Could not compute correlation.", level="warning")

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
        pre_indices = torch.where(mask_pre > threshold)[0].cpu().numpy()
        if min_count_pre is not None and len(pre_indices) < min_count_pre:
            self._log(f"Warning: Only {len(pre_indices)} pre-activation parameters above threshold. Selecting top-{min_count_pre} instead.", level="warning")
            values, top_indices = torch.topk(mask_pre, k=min_count_pre)
            pre_indices = top_indices.cpu().numpy()

        # Get indices for input features mask
        input_indices = torch.where(mask_input > threshold)[0].cpu().numpy()
        if min_count_input is not None and len(input_indices) < min_count_input:
            self._log(f"Warning: Only {len(input_indices)} input feature parameters above threshold. Selecting top-{min_count_input} instead.", level="warning")
            values, top_indices = torch.topk(mask_input, k=min_count_input)
            input_indices = top_indices.cpu().numpy()

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