import torch
import torch.nn as nn
import math
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from tqdm import tqdm
import os


class LoraModule(nn.Module):
    """
    A LoRA module implementation for gradient projection.

    This module adds a three-layer LoRA structure to an existing module:
    1. An encoder that projects the input
    2. A bottleneck layer (initialized to zero)
    3. A decoder that projects back

    The bottleneck ensures the forward and backward passes remain unchanged,
    while allowing access to the projected gradients.
    """

    def __init__(self,
                base_module: nn.Module,
                rank: int = 64,
                init_method: str = "random"):
        """
        Initialize the LoRA module.

        Args:
            base_module: The module to wrap with LoRA
            rank: The rank of the LoRA decomposition
            init_method: Initialization method ("random" or "pca")
        """
        super().__init__()

        self.base_module = base_module
        self.rank = rank
        self.init_method = init_method

        # Get the device from the base module
        self.device = next(base_module.parameters()).device

        # Determine dimensions based on module type
        if isinstance(base_module, nn.Linear):
            in_features = base_module.in_features
            out_features = base_module.out_features

            # Create LoRA layers
            self.lora_A = nn.Linear(in_features, rank, bias=False)  # Encoder
            self.lora_B = nn.Linear(rank, rank, bias=False)        # Bottleneck
            self.lora_C = nn.Linear(rank, out_features, bias=False) # Decoder

            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)  # Initialize bottleneck to zero
            nn.init.kaiming_uniform_(self.lora_C.weight, a=math.sqrt(5))

        elif isinstance(base_module, nn.Conv2d):
            in_channels = base_module.in_channels
            out_channels = base_module.out_channels
            kernel_size = base_module.kernel_size
            stride = base_module.stride
            padding = base_module.padding

            # Create LoRA layers
            self.lora_A = nn.Conv2d(
                in_channels, rank, kernel_size, stride, padding, bias=False
            )
            self.lora_B = nn.Conv2d(rank, rank, 1, bias=False)
            self.lora_C = nn.Conv2d(rank, out_channels, 1, bias=False)

            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)  # Initialize bottleneck to zero
            nn.init.kaiming_uniform_(self.lora_C.weight, a=math.sqrt(5))

        else:
            raise ValueError(f"Unsupported module type: {type(base_module)}")

        # Move LoRA layers to the same device as base module
        self.lora_A = self.lora_A.to(self.device)
        self.lora_B = self.lora_B.to(self.device)
        self.lora_C = self.lora_C.to(self.device)

        # Store intermediate activations
        self.lora_B_output = None
        self.projected_input = None
        self.projected_grad_pre_activation = None
        self.projected_grad = None

    def forward(self, x):
        """
        Forward pass through the LoRA module.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Base module forward pass
        base_output = self.base_module(x)

        # LoRA forward pass
        self.projected_input = self.lora_A(x)

        # Bottleneck (zero-initialized, so doesn't affect output)
        self.lora_B_output = self.lora_B(self.projected_input)
        self.lora_B_output.retain_grad()

        # Decoder
        lora_C_output = self.lora_C(self.lora_B_output)

        return base_output + lora_C_output

    def update_projected_grad(self):
        """Update the projected gradient using the output of lora_B."""
        if self.lora_B_output is not None and self.lora_B_output.grad is not None:
            self.projected_grad_pre_activation = self.lora_B_output.grad
            bsz = self.projected_input.shape[0]
            self.projected_grad = torch.einsum('ijk,ijl->ikl', self.lora_B_output.grad, self.projected_input).reshape(bsz, -1)

    def init_pca(self, covariance_data):
        """
        Initialize LoRA weights using PCA.

        Args:
            covariance_data: Covariance data for PCA initialization
        """
        if "forward" not in covariance_data or "backward" not in covariance_data:
            raise ValueError("Covariance data must contain 'forward' and 'backward' keys")

        # Compute SVD of forward covariance
        fwd_cov = covariance_data["forward"]
        U_fwd, S_fwd, _ = torch.svd(fwd_cov)
        top_fwd_vectors = U_fwd[:, :self.rank]

        # Compute SVD of backward covariance
        bwd_cov = covariance_data["backward"]
        U_bwd, S_bwd, _ = torch.svd(bwd_cov)
        top_bwd_vectors = U_bwd[:, :self.rank]

        # Update LoRA weights
        if isinstance(self.base_module, nn.Linear):
            self.lora_A.weight.data.copy_(top_fwd_vectors.t())
            self.lora_C.weight.data.copy_(top_bwd_vectors)
        elif isinstance(self.base_module, nn.Conv2d):
            # Reshape for convolutional layers
            self.lora_A.weight.data.copy_(
                top_fwd_vectors.t().view(self.lora_A.weight.shape)
            )
            self.lora_C.weight.data.copy_(
                top_bwd_vectors.view(self.lora_C.weight.shape)
            )


class LoraInfluence:
    """
    Influence function calculator that uses LoRA for gradient projection.

    This class implements efficient influence function calculation using
    LoRA projection to reduce the dimensionality of gradients.
    """

    def __init__(self,
                model: nn.Module,
                layer_type: str = "Linear",
                rank: int = 64,
                hessian: str = "kfac",
                init_method: str = "random",
                project_name: str = "lora_influence",
                save_dir: str = "./influence_results",
                cpu_offload: bool = False,
                label_key: str = "input_ids"):
        """
        Initialize the LoraInfluence calculator.

        Args:
            model: PyTorch model
            layer_type: Type of layers to add LoRA to
            rank: Rank for LoRA projection
            hessian: Method for Hessian approximation ("none", "kfac", "ekfac")
            init_method: Method for initializing LoRA weights
            project_name: Name for the project
            save_dir: Directory to save results
            cpu_offload: Whether to offload gradients to CPU
            label_key: Key for labels in batch data
        """
        self.model = model
        self.layer_type = layer_type
        self.rank = rank
        self.hessian = hessian
        self.init_method = init_method
        self.project_name = project_name
        self.save_dir = save_dir
        self.cpu_offload = cpu_offload
        self.label_key = label_key

        # Map layer types to PyTorch classes
        self.type_map = {
            "Linear": nn.Linear,
            "Conv2d": nn.Conv2d
        }

        # Store references to original and LoRA modules
        self.original_modules = {}
        self.lora_modules = {}

        # Storage for training data gradients
        self.train_gradients = []

        # Storage for covariance matrices and eigendecompositions
        self.covariance = {}
        self.eigenvalues = {}
        self.eigenvectors = {}
        self.ekfac_eigenvalues = {}

        # Add LoRA modules to the model
        self._add_lora_to_model()

        print(f"Initialized LoraInfluence with {len(self.lora_modules)} {layer_type} layers at rank {rank}")

    def _add_lora_to_model(self):
        """Add LoRA modules to the model."""
        target_type = self.type_map.get(self.layer_type)
        if target_type is None:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")

        # Find and replace modules with LoRA modules
        for name, module in list(self.model.named_modules()):
            # Skip non-leaf modules
            if len(list(module.children())) > 0:
                continue

            # Skip non-target modules
            if not isinstance(module, target_type):
                continue

            # Already a LoRA module
            if isinstance(module, LoraModule):
                self.lora_modules[name] = module
                continue

            # Get parent module and child name
            parent_name, child_name = self._get_parent_and_child(name)
            parent = self.model
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)

            # Create LoRA module
            lora_module = LoraModule(
                base_module=module,
                rank=self.rank,
                init_method=self.init_method
            )

            # Store original module
            self.original_modules[name] = module

            # Replace with LoRA module
            setattr(parent, child_name, lora_module)
            self.lora_modules[name] = lora_module

    def _get_parent_and_child(self, name):
        """
        Get parent module name and child attribute name.

        Args:
            name: Full module name

        Returns:
            Tuple of (parent_name, child_name)
        """
        parts = name.split('.')
        if len(parts) == 1:
            return '', parts[0]
        return '.'.join(parts[:-1]), parts[-1]

    def extract_training_data(self, train_dataloader, compute_hessian=True):
        """
        Efficiently extract all necessary information from training data in a single pass.
        This method computes covariance matrices, collects gradients, and optionally sets
        up Hessian approximation, all in one pass through the training data.

        Args:
            train_dataloader: DataLoader for training data
            compute_hessian: Whether to compute Hessian approximation

        Returns:
            Dict with covariance matrices and gradients
        """
        print("Extracting information from training data...")

        # Set model to eval mode
        self.model.eval()

        # Create mapping for module indices
        module_names = list(self.lora_modules.keys())
        self.module_to_idx = {name: idx for idx, name in enumerate(module_names)}
        self.module_names = module_names

        # Initialize accumulators for both covariance and gradients
        forward_accs = {name: [] for name in module_names}
        backward_accs = {name: [] for name in module_names}
        per_module_gradients = {name: [] for name in module_names}
        n_samples = 0

        # Process each batch
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Processing training data")):
            # Prepare inputs
            if isinstance(batch, dict):
                inputs = {k: v.to(self.model.device) for k, v in batch.items()}
            else:
                inputs = batch[0].to(self.model.device)

            # Forward pass
            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

            # Compute loss
            assert hasattr(outputs, "loss")
            logp = -outputs.loss
            loss = logp - torch.log(1 - torch.exp(logp))

            # Backward pass
            loss.backward()

            # Collect data for both covariance and gradients
            for name, module in self.lora_modules.items():
                # For covariance computation
                if module.projected_input is not None:
                    if compute_hessian:
                        # Store for covariance calculation
                        forward_accs[name].append(module.projected_input.detach())

                # Update and collect projected gradients
                module.update_projected_grad()

                if module.projected_grad_pre_activation is not None:
                    if compute_hessian:
                        # Store for covariance calculation
                        backward_accs[name].append(module.projected_grad_pre_activation.detach())

                    # Store gradients for influence computation
                    grad_tensor = module.projected_grad
                    if self.cpu_offload:
                        grad_tensor = grad_tensor.cpu()
                    per_module_gradients[name].append(grad_tensor)

            # Update sample count
            batch_size = inputs[self.label_key].size(0) if isinstance(inputs, dict) else inputs.size(0)
            n_samples += batch_size

            # Zero gradients
            self.model.zero_grad()

        print(f"Processed {n_samples} training samples")

        # Compute covariance matrices if needed
        if compute_hessian:
            covariance = {}

            for name in module_names:
                if not forward_accs[name] or not backward_accs[name]:
                    continue

                # Get first tensor to determine shape
                sample_fwd = forward_accs[name][0]
                sample_bwd = backward_accs[name][0]

                # Initialize covariance matrices
                fwd_dim = sample_fwd.shape[-1]  # Rank dimension
                bwd_dim = sample_bwd.shape[-1]  # Rank dimension

                device = "cpu" if self.cpu_offload else sample_fwd.device
                fwd_cov = torch.zeros((fwd_dim, fwd_dim), device=device)
                bwd_cov = torch.zeros((bwd_dim, bwd_dim), device=device)

                # Incrementally compute covariance
                processed_samples = 0

                for batch_idx in range(len(forward_accs[name])):
                    # Get batch data
                    fwd_batch = forward_accs[name][batch_idx]
                    bwd_batch = backward_accs[name][batch_idx]

                    # Flatten sequence dimension
                    fwd_flat = fwd_batch.reshape(-1, fwd_dim)
                    bwd_flat = bwd_batch.reshape(-1, bwd_dim)

                    # Compute covariance contribution
                    if self.cpu_offload:
                        # Use GPU for computation, then move back to CPU
                        fwd_cov_gpu = fwd_cov.to(device=fwd_batch.device)
                        fwd_cov_gpu.addmm_(fwd_flat.t(), fwd_flat)
                        fwd_cov = fwd_cov_gpu.to(device="cpu", non_blocking=True)

                        bwd_cov_gpu = bwd_cov.to(device=bwd_batch.device)
                        bwd_cov_gpu.addmm_(bwd_flat.t(), bwd_flat)
                        bwd_cov = bwd_cov_gpu.to(device="cpu", non_blocking=True)
                    else:
                        fwd_cov.addmm_(fwd_flat.t(), fwd_flat)
                        bwd_cov.addmm_(bwd_flat.t(), bwd_flat)

                    # Update processed samples count
                    processed_samples += fwd_flat.shape[0]

                # Normalize by total number of samples (including sequence positions)
                fwd_cov /= processed_samples
                bwd_cov /= processed_samples

                covariance[name] = {
                    "forward": fwd_cov,
                    "backward": bwd_cov
                }

            # Store covariance matrices
            self.covariance = covariance
            print(f"Computed covariance matrices for {len(covariance)} modules")

            # Initialize LoRA weights using PCA if requested
            if self.init_method == "pca":
                self._init_lora_from_pca(covariance)

            # Compute eigendecomposition for Hessian approximation
            if self.hessian in ["kfac", "ekfac"]:
                self._compute_eigendecomposition()

        # Concatenate gradients for each module
        concatenated_gradients = {}
        for name, grad_list in per_module_gradients.items():
            if grad_list:
                # Concatenate along the batch dimension (dim=0)
                concatenated_gradients[name] = torch.cat(grad_list, dim=0)

        # Get the gradient shapes for each module
        gradient_shapes = {name: tensor.shape[1:] for name, tensor in concatenated_gradients.items()}

        # Create the final tensor with shape (num_layers, num_samples, *per_sample_gradient_shape)
        # First, check if all gradient shapes are the same to use a single tensor
        shapes_are_uniform = len(set(str(shape) for shape in gradient_shapes.values())) == 1

        if shapes_are_uniform and concatenated_gradients:
            # All modules have the same gradient shape, create a single stacked tensor
            num_layers = len(module_names)
            num_samples = next(iter(concatenated_gradients.values())).shape[0]
            per_sample_shape = next(iter(gradient_shapes.values()))

            # Initialize the final tensor
            device = next(iter(concatenated_gradients.values())).device
            gradients = torch.zeros((num_layers, num_samples) + per_sample_shape, device=device)

            # Fill the tensor with gradients
            for name, tensor in concatenated_gradients.items():
                module_idx = self.module_to_idx[name]
                gradients[module_idx] = tensor

            print(f"Final gradient tensor shape: {gradients.shape}")
        else:
            # Gradient shapes differ, keep as dictionary with module indices
            gradients = {self.module_to_idx[name]: tensor for name, tensor in concatenated_gradients.items()}
            print(f"Created gradient dictionary with {len(gradients)} modules")

        # Store gradients
        self.train_gradients = gradients

        # Compute EKFAC eigenvalues if needed
        if compute_hessian and self.hessian == "ekfac":
            self._compute_ekfac_eigenvalues_from_gradients()

        return {
            "covariance": self.covariance if compute_hessian else None,
            "gradients": self.train_gradients
        }

    def _compute_eigendecomposition(self):
        """
        Compute eigendecomposition of covariance matrices with improved numerical stability.

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        print("Computing eigendecomposition of covariance matrices...")

        eigenvalues = {}
        eigenvectors = {}

        for name, cov_data in self.covariance.items():
            eigenvalues[name] = {}
            eigenvectors[name] = {}

            for cov_type in ["forward", "backward"]:
                cov = cov_data[cov_type]

                # Add a small regularization term to ensure positive definiteness
                # and improve numerical stability
                eps = 1e-6 * torch.eye(cov.shape[0], device=cov.device)
                stabilized_cov = cov + eps

                try:
                    # Try using eigh first (for symmetric matrices)
                    eigvals, eigvecs = torch.linalg.eigh(stabilized_cov)

                    # Sort in descending order (eigh returns in ascending)
                    idx = torch.argsort(eigvals, descending=True)
                    eigvals = eigvals[idx]
                    eigvecs = eigvecs[:, idx]

                except RuntimeError:
                    print(f"Falling back to SVD for {name}/{cov_type} due to eigh failure")
                    # Fall back to SVD which is more stable but slower
                    U, S, Vh = torch.linalg.svd(stabilized_cov, full_matrices=False)
                    eigvals = S  # Singular values = eigenvalues for PSD matrices
                    eigvecs = U  # Left singular vectors = eigenvectors for PSD matrices

                # Ensure eigenvalues are positive
                eigvals = torch.clamp(eigvals, min=1e-10)

                eigenvalues[name][cov_type] = eigvals
                eigenvectors[name][cov_type] = eigvecs

        print(f"Computed eigendecomposition for {len(eigenvalues)} modules")

        # Store the eigendecomposition
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

        return eigenvalues, eigenvectors

    def _compute_ekfac_eigenvalues_from_gradients(self):
        """
        Compute eigenvalue correction for EKFAC using previously collected gradients.
        This avoids another pass through the data.

        Returns:
            Dict of corrected eigenvalues
        """
        print("Computing EKFAC eigenvalue correction from gradients...")

        # Ensure eigendecomposition is already computed
        if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
            raise ValueError("Eigendecomposition must be computed before EKFAC correction")

        # Initialize EKFAC eigenvalues
        ekfac_eigenvalues = {}

        # Check if we have tensor or dictionary format for gradients
        using_tensor_format = isinstance(self.train_gradients, torch.Tensor)

        if using_tensor_format:
            for layer_idx, module_name in enumerate(self.module_names):
                if module_name not in self.eigenvectors:
                    continue

                # Get gradients for this layer
                gradients = self.train_gradients[layer_idx]

                # Get eigenvectors
                fwd_eigvec = self.eigenvectors[module_name]['forward'].to(device=gradients.device)
                bwd_eigvec = self.eigenvectors[module_name]['backward'].to(device=gradients.device)

                # Initialize eigenvalue accumulator
                ekfac_eigval = torch.zeros(
                    (len(bwd_eigvec), len(fwd_eigvec)),
                    device=gradients.device
                )

                # Process each gradient
                for i in range(gradients.shape[0]):
                    # Reshape gradient to match layer dimensions
                    grad_matrix = gradients[i].reshape(len(bwd_eigvec), -1)

                    # Rotate gradient using eigenvectors
                    rotated_grad = torch.matmul(
                        torch.matmul(bwd_eigvec.t(), grad_matrix),
                        fwd_eigvec
                    )

                    # Accumulate squared values
                    ekfac_eigval += rotated_grad.square()

                # Normalize by number of samples
                ekfac_eigval /= gradients.shape[0]

                # Store result
                ekfac_eigenvalues[module_name] = ekfac_eigval.cpu() if self.cpu_offload else ekfac_eigval
        else:
            # Dictionary format
            for module_idx, module_name in enumerate(self.module_names):
                if module_name not in self.eigenvectors or module_idx not in self.train_gradients:
                    continue

                # Get gradients for this module
                gradients = self.train_gradients[module_idx]

                # Get eigenvectors
                fwd_eigvec = self.eigenvectors[module_name]['forward'].to(device=gradients.device)
                bwd_eigvec = self.eigenvectors[module_name]['backward'].to(device=gradients.device)

                # Initialize eigenvalue accumulator
                ekfac_eigval = torch.zeros(
                    (len(bwd_eigvec), len(fwd_eigvec)),
                    device=gradients.device
                )

                # Process each gradient
                for i in range(gradients.shape[0]):
                    # Reshape gradient to match layer dimensions
                    grad_matrix = gradients[i].reshape(len(bwd_eigvec), -1)

                    # Rotate gradient using eigenvectors
                    rotated_grad = torch.matmul(
                        torch.matmul(bwd_eigvec.t(), grad_matrix),
                        fwd_eigvec
                    )

                    # Accumulate squared values
                    ekfac_eigval += rotated_grad.square()

                # Normalize by number of samples
                ekfac_eigval /= gradients.shape[0]

                # Store result
                ekfac_eigenvalues[module_name] = ekfac_eigval.cpu() if self.cpu_offload else ekfac_eigval

        print(f"Computed EKFAC eigenvalue correction for {len(ekfac_eigenvalues)} modules")

        # Store the EKFAC eigenvalues
        self.ekfac_eigenvalues = ekfac_eigenvalues

        return ekfac_eigenvalues

    def _init_lora_from_pca(self, covariance=None):
        """
        Initialize LoRA modules using PCA from covariance.

        Args:
            covariance: Dict of covariance matrices
        """
        print("Initializing LoRA modules from PCA...")

        if covariance is None:
            covariance = self.covariance

        if not covariance:
            raise ValueError("No covariance data available for PCA initialization")

        for name, cov_data in covariance.items():
            if name not in self.lora_modules:
                continue

            self.lora_modules[name].init_pca(cov_data)

    def collect_gradients(self, dataloader, dataset_type="train", max_batches=None):
        """
        Collect projected gradients from data.

        Args:
            dataloader: DataLoader for data
            dataset_type: Type of dataset ("train" or "test")
            max_batches: Maximum number of batches to process

        Returns:
            Tensor with shape (num_layers, num_samples, *per_sample_gradient_shape) for train
            or the same structure for test
        """
        print(f"Collecting projected gradients from {dataset_type} data...")
        # Set model to eval mode
        self.model.eval()

        # Create mapping for module indices
        module_names = list(self.lora_modules.keys())
        self.module_to_idx = {name: idx for idx, name in enumerate(module_names)}

        # Get total number of batches to process
        total_batches = min(len(dataloader), max_batches or float('inf'))

        # Initialize lists to collect gradients by module
        per_module_gradients = {name: [] for name in module_names}
        total_samples = 0

        # Process each batch
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {dataset_type} batches")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Prepare inputs
            if isinstance(batch, dict):
                inputs = {k: v.to(self.model.device) for k, v in batch.items()}
                labels = inputs.get(self.label_key)
            else:
                inputs = batch[0].to(self.model.device)
                labels = batch[1].to(self.model.device)

            # Forward pass
            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

            # Compute loss
            assert hasattr(outputs, "loss")
            logp = -outputs.loss
            loss = logp - torch.log(1 - torch.exp(logp))

            # Backward pass
            loss.backward()

            # Collect projected gradients
            for name, module in self.lora_modules.items():
                module.update_projected_grad()
                if module.projected_grad is not None:
                    grad_tensor = module.projected_grad
                    if self.cpu_offload:
                        grad_tensor = grad_tensor.cpu()
                    per_module_gradients[name].append(grad_tensor)

            # Update total sample count
            if per_module_gradients[module_names[0]]:  # Check if we have at least one gradient
                batch_size = per_module_gradients[module_names[0]][-1].shape[0]
                total_samples += batch_size

            # Zero gradients
            self.model.zero_grad()

        print(f"Collected gradients from {len(per_module_gradients[module_names[0]])} batches, total samples: {total_samples}")

        # Concatenate all batches for each module
        concatenated_gradients = {}
        for name, grad_list in per_module_gradients.items():
            if grad_list:
                # Concatenate along the batch dimension (dim=0)
                concatenated_gradients[name] = torch.cat(grad_list, dim=0)

        # Get the gradient shapes for each module
        gradient_shapes = {name: tensor.shape[1:] for name, tensor in concatenated_gradients.items()}

        # Create the final tensor with shape (num_layers, num_samples, *per_sample_gradient_shape)
        # First, check if all gradient shapes are the same to use a single tensor
        shapes_are_uniform = len(set(str(shape) for shape in gradient_shapes.values())) == 1

        if shapes_are_uniform and concatenated_gradients:
            # All modules have the same gradient shape, create a single stacked tensor
            num_layers = len(module_names)
            num_samples = next(iter(concatenated_gradients.values())).shape[0]
            per_sample_shape = next(iter(gradient_shapes.values()))

            # Initialize the final tensor
            device = next(iter(concatenated_gradients.values())).device
            gradients = torch.zeros((num_layers, num_samples) + per_sample_shape, device=device)

            # Fill the tensor with gradients
            for name, tensor in concatenated_gradients.items():
                module_idx = self.module_to_idx[name]
                gradients[module_idx] = tensor

            print(f"Final gradient tensor shape: {gradients.shape}")
        else:
            # Gradient shapes differ, keep as dictionary with module indices
            gradients = {self.module_to_idx[name]: tensor for name, tensor in concatenated_gradients.items()}
            print(f"Created gradient dictionary with {len(gradients)} modules")

        # Store module names for reference
        self.module_names = module_names

        # Store or return gradients based on dataset type
        if dataset_type == "train":
            self.train_gradients = gradients
            return self.train_gradients
        else:
            self.test_gradients = gradients
            return self.test_gradients

    def _precondition_gradients_kfac(self, gradients, module_name, damping=1e-5):
        """
        Precondition gradients using KFAC approximation.

        Args:
            gradients: Gradient tensor
            module_name: Name of the module
            damping: Damping parameter

        Returns:
            Preconditioned gradients
        """
        # Ensure eigendecomposition is already computed
        if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
            raise ValueError("Eigendecomposition must be computed before using KFAC")

        # Get eigenvectors and eigenvalues
        fwd_eigvec = self.eigenvectors[module_name]['forward']
        bwd_eigvec = self.eigenvectors[module_name]['backward']
        fwd_eigval = self.eigenvalues[module_name]['forward']
        bwd_eigval = self.eigenvalues[module_name]['backward']

        # Create full eigenvalue matrix (outer product)
        full_eigval = torch.outer(bwd_eigval, fwd_eigval)

        # Add damping
        full_eigval += damping

        # Reshape gradients to matrix form
        original_shape = gradients.shape
        gradients_2d = gradients.reshape(original_shape[0], -1)

        # Precondition gradients
        precond_grads = []
        for grad in gradients_2d:
            # Reshape to match the layer dimensions
            grad_matrix = grad.reshape(len(bwd_eigvec), -1)

            # Rotate gradients
            rotated_grad = torch.matmul(torch.matmul(bwd_eigvec.t(), grad_matrix), fwd_eigvec)

            # Precondition with inverse eigenvalues
            precond_rotated = rotated_grad / full_eigval

            # Rotate back
            precond_grad = torch.matmul(torch.matmul(bwd_eigvec, precond_rotated), fwd_eigvec.t())

            precond_grads.append(precond_grad.flatten())

        # Stack and reshape back to original shape
        return torch.stack(precond_grads).reshape(original_shape)

    def _precondition_gradients_ekfac(self, gradients, module_name, damping=1e-5):
        """
        Precondition gradients using EKFAC approximation.

        Args:
            gradients: Gradient tensor
            module_name: Name of the module
            damping: Damping parameter

        Returns:
            Preconditioned gradients
        """
        # Ensure EKFAC eigenvalues are already computed
        if not hasattr(self, 'ekfac_eigenvalues') or not self.ekfac_eigenvalues:
            raise ValueError("EKFAC eigenvalues must be computed before using EKFAC")

        # Get eigenvectors and EKFAC eigenvalues
        fwd_eigvec = self.eigenvectors[module_name]['forward']
        bwd_eigvec = self.eigenvectors[module_name]['backward']
        ekfac_eigval = self.ekfac_eigenvalues[module_name]

        # Add damping
        ekfac_eigval_damped = ekfac_eigval + damping

        # Reshape gradients to matrix form
        original_shape = gradients.shape
        gradients_2d = gradients.reshape(original_shape[0], -1)

        # Precondition gradients
        precond_grads = []
        for grad in gradients_2d:
            # Reshape to match the layer dimensions
            grad_matrix = grad.reshape(len(bwd_eigvec), -1)

            # Rotate gradients
            rotated_grad = torch.matmul(torch.matmul(bwd_eigvec.t(), grad_matrix), fwd_eigvec)

            # Precondition with inverse EKFAC eigenvalues
            precond_rotated = rotated_grad / ekfac_eigval_damped

            # Rotate back
            precond_grad = torch.matmul(torch.matmul(bwd_eigvec, precond_rotated), fwd_eigvec.t())

            precond_grads.append(precond_grad.flatten())

        # Stack and reshape back to original shape
        return torch.stack(precond_grads).reshape(original_shape)

#     def compute_influence(self, test_gradients=None, test_dataloader=None, damping=1e-5, max_test_batches=None):
#         """
#         Compute influence scores between test and training examples.

#         Args:
#             test_gradients: Precomputed test gradients (in tensor format)
#             test_dataloader: DataLoader for test data
#             damping: Damping parameter
#             max_test_batches: Maximum number of test batches to process

#         Returns:
#             Dict with influence scores
#         """
#         if self.train_gradients is None:
#             raise ValueError("No training gradients collected. Call collect_gradients first.")

#         # Get test gradients if not provided
#         if test_gradients is None:
#             if test_dataloader is None:
#                 raise ValueError("Either test_gradients or test_dataloader must be provided")
#             test_gradients = self.collect_gradients(
#                 test_dataloader,
#                 dataset_type="test",
#                 max_batches=max_test_batches
#             )

#         # Check if we're using tensor or dictionary format
#         using_tensor_format = isinstance(self.train_gradients, torch.Tensor) and isinstance(test_gradients, torch.Tensor)

#         print(f"Computing influence scores using {'tensor' if using_tensor_format else 'dictionary'} format...")

#         if using_tensor_format:
#             # Both gradients are tensors with shape (num_layers, num_samples, *per_sample_gradient_shape)
#             num_test_samples = test_gradients.shape[1]
#             num_train_samples = self.train_gradients.shape[1]

#             # Initialize influence matrix: [num_test_samples, num_train_samples]
#             influence_matrix = torch.zeros((num_test_samples, num_train_samples),
#                                         device=self.train_gradients.device)

#             # Generate sample IDs
#             test_sample_ids = [f"test_sample_{i}" for i in range(num_test_samples)]
#             train_sample_ids = [f"train_sample_{i}" for i in range(num_train_samples)]

#             # Compute influence scores based on hessian approximation
#             if self.hessian == "none":
#                 # For each test sample
#                 for test_idx in tqdm(range(num_test_samples), desc="Computing influence scores"):
#                     # For each train sample
#                     for train_idx in range(num_train_samples):
#                         # Sum dot products across all layers
#                         dot_sum = 0

#                         # For each layer
#                         for layer_idx in range(test_gradients.shape[0]):
#                             test_grad = test_gradients[layer_idx, test_idx]
#                             train_grad = self.train_gradients[layer_idx, train_idx]

#                             # Flatten and compute dot product
#                             dot_product = torch.dot(test_grad.flatten(), train_grad.flatten())
#                             dot_sum += dot_product.item()

#                         # Store the influence score
#                         influence_matrix[test_idx, train_idx] = dot_sum

#             elif self.hessian == "kfac":
#                 # First ensure we have eigendecomposition computed
#                 if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
#                     print("Computing eigendecomposition for KFAC...")
#                     self._compute_eigendecomposition()

#                 # Process each layer separately with KFAC preconditioning
#                 for layer_idx in range(test_gradients.shape[0]):
#                     module_name = self.module_names[layer_idx]

#                     # Skip if we don't have eigendecomposition for this module
#                     if module_name not in self.eigenvectors:
#                         continue

#                     # Extract gradients for this layer
#                     test_layer_grads = test_gradients[layer_idx]
#                     train_layer_grads = self.train_gradients[layer_idx]

#                     # Precondition the train gradients
#                     precond_train_grads = self._precondition_gradients_kfac(
#                         train_layer_grads,
#                         module_name,
#                         damping
#                     )

#                     # For each test sample
#                     for test_idx in tqdm(range(num_test_samples), desc=f"Computing KFAC influence for layer {layer_idx}"):
#                         test_grad = test_layer_grads[test_idx]

#                         # For each train sample
#                         for train_idx in range(num_train_samples):
#                             precond_train_grad = precond_train_grads[train_idx]

#                             # Compute dot product with preconditioned gradient
#                             dot_product = torch.dot(test_grad.flatten(), precond_train_grad.flatten())

#                             # Accumulate influence
#                             influence_matrix[test_idx, train_idx] += dot_product.item()

#             elif self.hessian == "ekfac":
#                 # First ensure we have eigendecomposition and EKFAC eigenvalues computed
#                 if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
#                     print("Computing eigendecomposition for EKFAC...")
#                     self._compute_eigendecomposition()

#                 if not hasattr(self, 'ekfac_eigenvalues') or not self.ekfac_eigenvalues:
#                     raise ValueError("EKFAC eigenvalues must be computed before using EKFAC. Run _compute_ekfac_eigenvalues first.")

#                 # Process each layer separately with EKFAC preconditioning
#                 for layer_idx in range(test_gradients.shape[0]):
#                     module_name = self.module_names[layer_idx]

#                     # Skip if we don't have EKFAC data for this module
#                     if module_name not in self.ekfac_eigenvalues:
#                         continue

#                     # Extract gradients for this layer
#                     test_layer_grads = test_gradients[layer_idx]
#                     train_layer_grads = self.train_gradients[layer_idx]

#                     # Precondition the train gradients
#                     precond_train_grads = self._precondition_gradients_ekfac(
#                         train_layer_grads,
#                         module_name,
#                         damping
#                     )

#                     # For each test sample
#                     for test_idx in tqdm(range(num_test_samples), desc=f"Computing EKFAC influence for layer {layer_idx}"):
#                         test_grad = test_layer_grads[test_idx]

#                         # For each train sample
#                         for train_idx in range(num_train_samples):
#                             precond_train_grad = precond_train_grads[train_idx]

#                             # Compute dot product with preconditioned gradient
#                             dot_product = torch.dot(test_grad.flatten(), precond_train_grad.flatten())

#                             # Accumulate influence
#                             influence_matrix[test_idx, train_idx] += dot_product.item()

#             else:
#                 raise ValueError(f"Unsupported hessian approximation: {self.hessian}")

#             # Create result dictionary
#             result = {
#                 "src_ids": test_sample_ids,
#                 "tgt_ids": train_sample_ids,
#                 "influence": influence_matrix
#             }

#         else:
#             # Dictionary format (for non-uniform gradient shapes)
#             # In this case, both test_gradients and self.train_gradients are dictionaries
#             # with module indices as keys and tensors as values

#             # Get all module indices that are in both test and train gradients
#             common_modules = set(test_gradients.keys()) & set(self.train_gradients.keys())

#             if not common_modules:
#                 raise ValueError("No common modules found between test and train gradients")

#             # Get dimensions
#             test_module_idx = list(common_modules)[0]
#             train_module_idx = list(common_modules)[0]

#             num_test_samples = test_gradients[test_module_idx].shape[0]
#             num_train_samples = self.train_gradients[train_module_idx].shape[0]

#             # Initialize influence matrix
#             influence_matrix = torch.zeros((num_test_samples, num_train_samples),
#                                         device=test_gradients[test_module_idx].device)

#             # Generate sample IDs
#             test_sample_ids = [f"test_sample_{i}" for i in range(num_test_samples)]
#             train_sample_ids = [f"train_sample_{i}" for i in range(num_train_samples)]

#             # Compute influence scores based on hessian approximation
#             if self.hessian == "none":
#                 # For each test sample
#                 for test_idx in tqdm(range(num_test_samples), desc="Computing influence scores"):
#                     # For each train sample
#                     for train_idx in range(num_train_samples):
#                         # Sum dot products across all common modules
#                         dot_sum = 0

#                         # For each common module
#                         for module_idx in common_modules:
#                             test_grad = test_gradients[module_idx][test_idx]
#                             train_grad = self.train_gradients[module_idx][train_idx]

#                             # Flatten and compute dot product
#                             dot_product = torch.dot(test_grad.flatten(), train_grad.flatten())
#                             dot_sum += dot_product.item()

#                         # Store the influence score
#                         influence_matrix[test_idx, train_idx] = dot_sum

#             elif self.hessian == "kfac":
#                 # First ensure we have eigendecomposition computed
#                 if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
#                     print("Computing eigendecomposition for KFAC...")
#                     self._compute_eigendecomposition()

#                 # For each common module
#                 for module_idx in common_modules:
#                     module_name = self.module_names[module_idx]

#                     # Skip if we don't have eigendecomposition for this module
#                     if module_name not in self.eigenvectors:
#                         continue

#                     # Extract gradients for this module
#                     test_module_grads = test_gradients[module_idx]
#                     train_module_grads = self.train_gradients[module_idx]

#                     # Precondition the train gradients
#                     precond_train_grads = self._precondition_gradients_kfac(
#                         train_module_grads,
#                         module_name,
#                         damping
#                     )

#                     # For each test sample
#                     for test_idx in tqdm(range(num_test_samples), desc=f"Computing KFAC influence for module {module_idx}"):
#                         test_grad = test_module_grads[test_idx]

#                         # For each train sample
#                         for train_idx in range(num_train_samples):
#                             precond_train_grad = precond_train_grads[train_idx]

#                             # Compute dot product with preconditioned gradient
#                             dot_product = torch.dot(test_grad.flatten(), precond_train_grad.flatten())

#                             # Accumulate influence
#                             influence_matrix[test_idx, train_idx] += dot_product.item()

#             elif self.hessian == "ekfac":
#                 # First ensure we have eigendecomposition and EKFAC eigenvalues computed
#                 if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
#                     print("Computing eigendecomposition for EKFAC...")
#                     self._compute_eigendecomposition()

#                 if not hasattr(self, 'ekfac_eigenvalues') or not self.ekfac_eigenvalues:
#                     raise ValueError("EKFAC eigenvalues must be computed before using EKFAC. Run _compute_ekfac_eigenvalues first.")

#                 # For each common module
#                 for module_idx in common_modules:
#                     module_name = self.module_names[module_idx]

#                     # Skip if we don't have EKFAC data for this module
#                     if module_name not in self.ekfac_eigenvalues:
#                         continue

#                     # Extract gradients for this module
#                     test_module_grads = test_gradients[module_idx]
#                     train_module_grads = self.train_gradients[module_idx]

#                     # Precondition the train gradients
#                     precond_train_grads = self._precondition_gradients_ekfac(
#                         train_module_grads,
#                         module_name,
#                         damping
#                     )

#                     # For each test sample
#                     for test_idx in tqdm(range(num_test_samples), desc=f"Computing EKFAC influence for module {module_idx}"):
#                         test_grad = test_module_grads[test_idx]

#                         # For each train sample
#                         for train_idx in range(num_train_samples):
#                             precond_train_grad = precond_train_grads[train_idx]

#                             # Compute dot product with preconditioned gradient
#                             dot_product = torch.dot(test_grad.flatten(), precond_train_grad.flatten())

#                             # Accumulate influence
#                             influence_matrix[test_idx, train_idx] += dot_product.item()

#             else:
#                 raise ValueError(f"Unsupported hessian approximation: {self.hessian}")

#             # Create result dictionary
#             result = {
#                 "src_ids": test_sample_ids,
#                 "tgt_ids": train_sample_ids,
#                 "influence": influence_matrix
#             }

#         print(f"Computed influence scores. Shape: {influence_matrix.shape}")

#         return result

    def compute_influence(self, test_gradients=None, test_dataloader=None, damping=1e-5, max_test_batches=None):
        """
        Compute influence scores between test and training examples efficiently using matrix operations.

        Args:
            test_gradients: Precomputed test gradients (in tensor format)
            test_dataloader: DataLoader for test data
            damping: Damping parameter
            max_test_batches: Maximum number of test batches to process

        Returns:
            Dict with influence scores
        """
        if self.train_gradients is None:
            raise ValueError("No training gradients collected. Call collect_gradients first.")

        # Get test gradients if not provided
        if test_gradients is None:
            if test_dataloader is None:
                raise ValueError("Either test_gradients or test_dataloader must be provided")
            test_gradients = self.collect_gradients(
                test_dataloader,
                dataset_type="test",
                max_batches=max_test_batches
            )

        # Check if we're using tensor or dictionary format
        using_tensor_format = isinstance(self.train_gradients, torch.Tensor) and isinstance(test_gradients, torch.Tensor)

        print(f"Computing influence scores using {'tensor' if using_tensor_format else 'dictionary'} format...")

        if using_tensor_format:
            # Both gradients are tensors with shape (num_layers, num_samples, *per_sample_gradient_shape)
            num_test_samples = test_gradients.shape[1]
            num_train_samples = self.train_gradients.shape[1]

            # Initialize influence matrix: [num_test_samples, num_train_samples]
            influence_matrix = torch.zeros((num_test_samples, num_train_samples),
                                        device=self.train_gradients.device)

            # Generate sample IDs
            test_sample_ids = [f"test_sample_{i}" for i in range(num_test_samples)]
            train_sample_ids = [f"train_sample_{i}" for i in range(num_train_samples)]

            # Compute influence scores based on hessian approximation
            if self.hessian == "none":
                # For each layer
                for layer_idx in range(test_gradients.shape[0]):
                    # Get test gradients for this layer [num_test_samples, feature_dims]
                    test_layer_grads = test_gradients[layer_idx].reshape(num_test_samples, -1)

                    # Get train gradients for this layer [num_train_samples, feature_dims]
                    train_layer_grads = self.train_gradients[layer_idx].reshape(num_train_samples, -1)

                    # Compute influence via matrix multiplication: [num_test_samples, num_train_samples]
                    # test_grads @ train_grads.T gives the dot product between each pair
                    layer_influence = torch.matmul(train_layer_grads, test_layer_grads.t())

                    # Add to total influence
                    influence_matrix += layer_influence

            elif self.hessian == "kfac":
                # First ensure we have eigendecomposition computed
                if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
                    print("Computing eigendecomposition for KFAC...")
                    self._compute_eigendecomposition()

                # Process each layer separately with KFAC preconditioning
                for layer_idx in range(test_gradients.shape[0]):
                    module_name = self.module_names[layer_idx]

                    # Skip if we don't have eigendecomposition for this module
                    if module_name not in self.eigenvectors:
                        continue

                    # Extract gradients for this layer
                    test_layer_grads = test_gradients[layer_idx].reshape(num_test_samples, -1)
                    train_layer_grads = self.train_gradients[layer_idx].reshape(num_train_samples, -1)

                    # Precondition the train gradients (this is the correct approach)
                    precond_train_grads = self._precondition_gradients_kfac(
                        self.train_gradients[layer_idx],
                        module_name,
                        damping
                    ).reshape(num_train_samples, -1)

                    # Compute influence via matrix multiplication
                    layer_influence = torch.matmul(precond_train_grads, test_layer_grads.t())

                    # Add to total influence
                    influence_matrix += layer_influence

            elif self.hessian == "ekfac":
                # First ensure we have eigendecomposition and EKFAC eigenvalues computed
                if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
                    print("Computing eigendecomposition for EKFAC...")
                    self._compute_eigendecomposition()

                if not hasattr(self, 'ekfac_eigenvalues') or not self.ekfac_eigenvalues:
                    raise ValueError("EKFAC eigenvalues must be computed before using EKFAC. Run _compute_ekfac_eigenvalues first.")

                # Process each layer separately with EKFAC preconditioning
                for layer_idx in range(test_gradients.shape[0]):
                    module_name = self.module_names[layer_idx]

                    # Skip if we don't have EKFAC data for this module
                    if module_name not in self.ekfac_eigenvalues:
                        continue

                    # Extract gradients for this layer
                    test_layer_grads = test_gradients[layer_idx].reshape(num_test_samples, -1)
                    train_layer_grads = self.train_gradients[layer_idx].reshape(num_train_samples, -1)

                    # Precondition the train gradients (this is the correct approach)
                    precond_train_grads = self._precondition_gradients_ekfac(
                        self.train_gradients[layer_idx],
                        module_name,
                        damping
                    ).reshape(num_train_samples, -1)

                    # Compute influence via matrix multiplication
                    layer_influence = torch.matmul(precond_train_grads, test_layer_grads.t())

                    # Add to total influence
                    influence_matrix += layer_influence

            else:
                raise ValueError(f"Unsupported hessian approximation: {self.hessian}")

            # Create result dictionary
            result = {
                "src_ids": test_sample_ids,
                "tgt_ids": train_sample_ids,
                "influence": influence_matrix
            }

        else:
            # Dictionary format (for non-uniform gradient shapes)
            # In this case, both test_gradients and self.train_gradients are dictionaries
            # with module indices as keys and tensors as values

            # Get all module indices that are in both test and train gradients
            common_modules = set(test_gradients.keys()) & set(self.train_gradients.keys())

            if not common_modules:
                raise ValueError("No common modules found between test and train gradients")

            # Get dimensions
            test_module_idx = list(common_modules)[0]
            train_module_idx = list(common_modules)[0]

            num_test_samples = test_gradients[test_module_idx].shape[0]
            num_train_samples = self.train_gradients[train_module_idx].shape[0]

            # Initialize influence matrix
            device = test_gradients[test_module_idx].device
            influence_matrix = torch.zeros((num_test_samples, num_train_samples), device=device)

            # Generate sample IDs
            test_sample_ids = [f"test_sample_{i}" for i in range(num_test_samples)]
            train_sample_ids = [f"train_sample_{i}" for i in range(num_train_samples)]

            # Compute influence scores based on hessian approximation
            if self.hessian == "none":
                # For each common module
                for module_idx in common_modules:
                    # Get test gradients [num_test_samples, feature_dims]
                    test_module_grads = test_gradients[module_idx].reshape(num_test_samples, -1)

                    # Get train gradients [num_train_samples, feature_dims]
                    train_module_grads = self.train_gradients[module_idx].reshape(num_train_samples, -1)

                    # Compute influence via matrix multiplication
                    module_influence = torch.matmul(train_module_grads, test_module_grads.t())

                    # Add to total influence
                    influence_matrix += module_influence

            elif self.hessian == "kfac":
                # First ensure we have eigendecomposition computed
                if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
                    print("Computing eigendecomposition for KFAC...")
                    self._compute_eigendecomposition()

                # For each common module
                for module_idx in common_modules:
                    module_name = self.module_names[module_idx]

                    # Skip if we don't have eigendecomposition for this module
                    if module_name not in self.eigenvectors:
                        continue

                    # Extract gradients for this module
                    test_module_grads = test_gradients[module_idx].reshape(num_test_samples, -1)
                    train_module_grads = self.train_gradients[module_idx].reshape(num_train_samples, -1)

                    # Precondition the train gradients
                    precond_train_grads = self._precondition_gradients_kfac(
                        self.train_gradients[module_idx],
                        module_name,
                        damping
                    ).reshape(num_train_samples, -1)

                    # Compute influence via matrix multiplication
                    module_influence = torch.matmul(precond_train_grads, test_module_grads.t())

                    # Add to total influence
                    influence_matrix += module_influence

            elif self.hessian == "ekfac":
                # First ensure we have eigendecomposition and EKFAC eigenvalues computed
                if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
                    print("Computing eigendecomposition for EKFAC...")
                    self._compute_eigendecomposition()

                if not hasattr(self, 'ekfac_eigenvalues') or not self.ekfac_eigenvalues:
                    raise ValueError("EKFAC eigenvalues must be computed before using EKFAC. Run _compute_ekfac_eigenvalues first.")

                # For each common module
                for module_idx in common_modules:
                    module_name = self.module_names[module_idx]

                    # Skip if we don't have EKFAC data for this module
                    if module_name not in self.ekfac_eigenvalues:
                        continue

                    # Extract gradients for this module
                    test_module_grads = test_gradients[module_idx].reshape(num_test_samples, -1)
                    train_module_grads = self.train_gradients[module_idx].reshape(num_train_samples, -1)

                    # Precondition the train gradients
                    precond_train_grads = self._precondition_gradients_ekfac(
                        self.train_gradients[module_idx],
                        module_name,
                        damping
                    ).reshape(num_train_samples, -1)

                    # Compute influence via matrix multiplication
                    module_influence = torch.matmul(precond_train_grads, test_module_grads.t())

                    # Add to total influence
                    influence_matrix += module_influence

            else:
                raise ValueError(f"Unsupported hessian approximation: {self.hessian}")

            # Create result dictionary
            result = {
                "src_ids": test_sample_ids,
                "tgt_ids": train_sample_ids,
                "influence": influence_matrix
            }

        print(f"Computed influence scores. Shape: {influence_matrix.shape}")

        return result