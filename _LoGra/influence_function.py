from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
import time
from .lora.modules import LoraModule

class IFAttributor:
    """
    Influence function calculator that uses LoRA for gradient projection.

    This class implements efficient influence function calculation using
    LoRA projection to reduce the dimensionality of gradients.
    """

    def __init__(
            self,
            model: nn.Module,
            layer_name: str = "Linear",
            hessian: str = "kfac",
            damping: float = None,
            projector_kwargs: dict = None,
            profile: bool = False,
            cpu_offload: bool = False,
        ):
        """
        Initialize the LoraInfluence calculator.

        Args:
            model: PyTorch model
            layer_type: Type of layers to add LoRA to
            hessian: Method for Hessian approximation ("none", "raw", "kfac", "ekfac")
            projector_kwargs: Dictionary of projector configuration
            cpu_offload: Whether to offload gradients to CPU
        """
        self.model = model
        self.layer_type = layer_name
        self.hessian = hessian
        self.damping = damping
        self.profile = profile
        self.cpu_offload = cpu_offload

        proj_dim = projector_kwargs.get("proj_dim", None)
        method = projector_kwargs.get("method", None)
        proj_factorize = projector_kwargs.get("proj_factorize", True)

        projector_kwargs.pop("proj_factorize")
        projector_kwargs.pop("proj_dim")
        projector_kwargs.pop("method")

        assert proj_factorize, "Projection must be factorized for LoRA influence calculation"

        self.rank = proj_dim
        self.init_method = method

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
        self.cov = {}
        self.eigenvalues = {}
        self.eigenvectors = {}
        self.ekfac_eigenvalues = {}

        # Add LoRA modules to the model
        self._add_lora()
        self.lora_module_names = list(self.lora_modules.keys())
        self.lora_module_to_idx = {name: idx for idx, name in enumerate(self.lora_module_names)}

        print(f"Initialized LoraInfluence with {len(self.lora_modules)} {layer_name} layers at rank {self.rank}")

        if self.profile:
            self.profiling_stats = {
                'projection': 0.0,
                'forward': 0.0,
                'backward': 0.0,
                'precondition': 0.0,
            }

    def _add_lora(self):
        """Add LoRA modules to the model."""
        target_type = self.type_map.get(self.layer_type)
        if target_type is None:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")

        def get_parent_and_child(name):
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
            parent_name, child_name = get_parent_and_child(name)
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

    def _init_lora_from_pca(self, covariance=None):
        """
        Initialize LoRA modules using PCA from covariance.

        Args:
            covariance: Dict of covariance matrices
        """
        print("Initializing LoRA modules from PCA...")

        if covariance is None:
            covariance = self.cov

        if not covariance:
            raise ValueError("No covariance data available for PCA initialization")

        for name, cov_data in covariance.items():
            if name not in self.lora_modules:
                continue

            self.lora_modules[name].init_pca(cov_data)

    def cache(self, train_dataloader):
        """
        Efficiently extract all necessary information from training data in a single pass.
        This method computes covariance matrices, collects gradients, and optionally sets
        up Hessian approximation, all in one pass through the training data.

        Args:
            train_dataloader: DataLoader for training data

        Returns:
            Dict with covariance matrices and gradients
        """
        print("Extracting information from training data...")

        # Set model to eval mode
        self.model.eval()

        # Initialize accumulators for both covariance and gradients
        per_module_gradients = {name: [] for name in self.lora_module_names}
        if self.hessian in ["kfac", "ekfac"]:
            per_module_forward = {name: [] for name in self.lora_module_names}
            per_module_backward = {name: [] for name in self.lora_module_names}

        n_samples = 0

        # Process each batch
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Processing training data")):
            # Zero gradients
            self.model.zero_grad()

            # Prepare inputs
            if isinstance(batch, dict):
                inputs = {k: v.to(self.model.device) for k, v in batch.items()}
            else:
                inputs = batch[0].to(self.model.device)

            # Time forward pass
            if self.profile:
                torch.cuda.synchronize()
                start_time = time.time()

            # Forward pass
            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

            if self.profile:
                torch.cuda.synchronize()
                self.profiling_stats['forward'] += time.time() - start_time

            # Compute custom loss
            logp = -outputs.loss
            loss = logp - torch.log(1 - torch.exp(logp))

            # Time backward pass
            if self.profile:
                torch.cuda.synchronize()
                start_time = time.time()

            # Backward pass
            loss.backward(retain_graph=True)

            if self.profile:
                torch.cuda.synchronize()
                self.profiling_stats['backward'] += time.time() - start_time

            for name, module in self.lora_modules.items():
                # Update and collect projected gradients
                module.update_projected_grad()

                # Store gradients for influence computation
                grad_tensor = module.projected_grad.detach()
                if self.cpu_offload:
                    grad_tensor = grad_tensor.cpu()

                per_module_gradients[name].append(grad_tensor)

                # For covariance computation
                if self.hessian in ["kfac", "ekfac"]:
                    per_module_forward[name].append(module.projected_input.detach())
                    per_module_backward[name].append(module.projected_grad_pre_activation.detach())

        # Time hessian
        if self.profile:
            torch.cuda.synchronize()
            start_time = time.time()

        # Compute covariance matrices if needed
        if self.hessian == "raw":
            grad_covariance = {}

            for name in self.lora_module_names:
                if name not in per_module_gradients:
                    continue

                # Concatenate all gradient batches
                all_grads = torch.cat(per_module_gradients[name], dim=0)
                grad_dim = all_grads.shape[1]  # Feature dimension of gradients

                # Initialize covariance matrix
                device = "cpu" if self.cpu_offload else all_grads.device
                cov = torch.zeros((grad_dim, grad_dim), device=device)

                # Compute covariance incrementally to save memory
                batch_size = 16  # Process in small batches to avoid OOM
                num_samples = all_grads.shape[0]

                for i in range(0, num_samples, batch_size):
                    end_idx = min(i + batch_size, num_samples)
                    batch = all_grads[i:end_idx]

                    if self.cpu_offload:
                        # Use GPU for computation, then move back to CPU
                        cov_gpu = cov.to(device=batch.device)
                        cov_gpu.addmm_(batch.t(), batch)
                        cov = cov_gpu.to(device="cpu", non_blocking=True)
                    else:
                        cov.addmm_(batch.t(), batch)

                # Normalize by number of samples
                cov /= num_samples

                # Store covariance
                grad_covariance[name] = {
                    "grad": cov
                }

            self.cov = grad_covariance
            print(f"Computed gradient covariance for {len(grad_covariance)} modules")

        elif self.hessian in ["kfac", "ekfac"]:
            fwd_bwd_covariance = {}

            for name in self.lora_module_names:
                if not per_module_forward[name] or not per_module_backward[name]:
                    continue

                # Get first tensor to determine shape
                sample_fwd = per_module_forward[name][0]
                sample_bwd = per_module_backward[name][0]

                # Initialize covariance matrices
                fwd_dim = sample_fwd.shape[-1]  # Rank dimension
                bwd_dim = sample_bwd.shape[-1]  # Rank dimension

                device = "cpu" if self.cpu_offload else sample_fwd.device
                fwd_cov = torch.zeros((fwd_dim, fwd_dim), device=device)
                bwd_cov = torch.zeros((bwd_dim, bwd_dim), device=device)

                # Incrementally compute covariance
                processed_samples = 0

                for batch_idx in range(len(per_module_forward[name])):
                    # Get batch data
                    fwd_batch = per_module_forward[name][batch_idx]
                    bwd_batch = per_module_backward[name][batch_idx]

                    # Flatten sequence dimension
                    fwd_flat = fwd_batch.view(-1, fwd_dim)
                    bwd_flat = bwd_batch.view(-1, bwd_dim)

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

                fwd_bwd_covariance[name] = {
                    "forward": fwd_cov,
                    "backward": bwd_cov
                }

            # Store covariance matrices
            self.cov = fwd_bwd_covariance
            print(f"Computed forward/backward covariance matrices for {len(fwd_bwd_covariance)} modules")

            # Initialize LoRA weights using PCA if requested
            if self.init_method == "pca":
                self._init_lora_from_pca(fwd_bwd_covariance) #TODO: this is not intended: we need to redo the forward/backward again to take into account the new weights

            # Compute eigendecomposition for Hessian approximation
            if self.hessian in ["kfac", "ekfac"]:
                if self.profile:
                    torch.cuda.synchronize()
                    start_time = time.time()

                self._compute_eigendecomposition()

                if self.profile:
                    torch.cuda.synchronize()
                    self.profiling_stats['precondition'] += time.time() - start_time

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
            num_layers = len(self.lora_module_names)
            num_samples = next(iter(concatenated_gradients.values())).shape[0]
            per_sample_shape = next(iter(gradient_shapes.values()))

            # Initialize the final tensor
            device = next(iter(concatenated_gradients.values())).device
            gradients = torch.zeros((num_layers, num_samples) + per_sample_shape, device=device)

            # Fill the tensor with gradients
            for name, tensor in concatenated_gradients.items():
                module_idx = self.lora_module_to_idx[name]
                gradients[module_idx] = tensor

        else:
            # Gradient shapes differ, keep as dictionary with module indices
            gradients = {self.lora_module_to_idx[name]: tensor for name, tensor in concatenated_gradients.items()}
            print(f"Created gradient dictionary with {len(gradients)} modules")

        # Store gradients
        self.train_gradients = gradients

        if self.hessian == "ekfac":
            # Compute EK-FAC eigenvalues if needed
            if self.profile:
                torch.cuda.synchronize()
                start_time = time.time()

            self._compute_ekfac_eigenvalues_from_gradients()

            if self.profile:
                torch.cuda.synchronize()
                self.profiling_stats['precondition'] += time.time() - start_time

        return {
            "covariance": self.cov if self.hessian in ["raw", "kfac", "ekfac"] else None,
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

        for name, cov_data in self.cov.items():
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


    def _compute_grad_covariance_inverse(self):
        """
        Compute the inverse of gradient covariance for raw Hessian approximation.

        Returns:
            Dict of inverse covariance matrices
        """
        print("Computing gradient covariance inverse...")

        if not hasattr(self, 'covariance') or not self.cov:
            raise ValueError("Gradient covariance must be computed before computing inverse")

        # Initialize inverse covariance dict
        con_inv = {}

        for name, cov in self.cov.items():
            # Get gradient covariance
            grad_cov = cov.get("grad")

            if grad_cov is None:
                continue

            # Add damping for numerical stability
            if self.damping is None:
                self.damping = 0.1 * torch.trace(grad_cov) / grad_cov.size(0)

            # Add diagonal damping
            damped_cov = grad_cov + self.damping * torch.eye(grad_cov.size(0), device=grad_cov.device)

            # Compute inverse
            try:
                # Try Cholesky decomposition first (more stable)
                L = torch.linalg.cholesky(damped_cov)
                inverse = torch.cholesky_inverse(L)
            except RuntimeError:
                print(f"Falling back to direct inverse for {name} due to Cholesky failure")
                # Fall back to direct inverse
                inverse = torch.inverse(damped_cov)

            con_inv[name] = {
                "grad": inverse
            }

        print(f"Computed gradient covariance inverse for {len(con_inv)} modules")

        # Store inverse covariance
        self.cov_inv = con_inv

        return con_inv

    def _compute_ekfac_eigenvalues_from_gradients(self):
        """
        Compute eigenvalue correction for EK-FAC using previously collected gradients.
        This avoids another pass through the data.

        Returns:
            Dict of corrected eigenvalues
        """
        print("Computing EK-FAC eigenvalue correction from gradients...")

        # Ensure eigendecomposition is already computed
        if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
            raise ValueError("Eigendecomposition must be computed before EK-FAC correction")

        # Initialize EK-FAC eigenvalues
        ekfac_eigenvalues = {}

        for module_name in self.lora_module_names:
            if module_name not in self.eigenvectors:
                continue

            # Get gradients for this layer
            gradients = self.train_gradients[self.lora_module_to_idx[module_name]]

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

        print(f"Computed EK-FAC eigenvalue correction for {len(ekfac_eigenvalues)} modules")

        # Store the EK-FAC eigenvalues
        self.ekfac_eigenvalues = ekfac_eigenvalues

        return ekfac_eigenvalues


    def collect_gradients(self, dataloader, dataset_type="train"):
        """
        Collect projected gradients from data.

        Args:
            dataloader: DataLoader for data
            dataset_type: Type of dataset ("train" or "test")

        Returns:
            Tensor with shape (num_layers, num_samples, *per_sample_gradient_shape) for train
            or the same structure for test
        """
        print(f"Collecting projected gradients from {dataset_type} data...")
        # Set model to eval mode
        self.model.eval()

        # Initialize lists to collect gradients by module
        per_module_gradients = {name: [] for name in self.lora_module_names}
        total_samples = 0

        # Process each batch
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {dataset_type} batches")):
            # Zero gradients
            self.model.zero_grad()

            # Prepare inputs
            if isinstance(batch, dict):
                inputs = {k: v.to(self.model.device) for k, v in batch.items()}
            else:
                inputs = batch[0].to(self.model.device)

            # Time forward pass
            if self.profile:
                torch.cuda.synchronize()
                start_time = time.time()

            # Forward pass
            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

            if self.profile:
                torch.cuda.synchronize()
                self.profiling_stats['forward'] += time.time() - start_time

            logp = -outputs.loss
            loss = logp - torch.log(1 - torch.exp(logp))

            # Time backward pass
            if self.profile:
                torch.cuda.synchronize()
                start_time = time.time()

            # Backward pass
            loss.backward()

            if self.profile:
                torch.cuda.synchronize()
                self.profiling_stats['backward'] += time.time() - start_time

            # Collect projected gradients
            for name, module in self.lora_modules.items():
                module.update_projected_grad()
                grad_tensor = module.projected_grad.detach()
                if self.cpu_offload:
                    grad_tensor = grad_tensor.cpu()
                per_module_gradients[name].append(grad_tensor)

            # Update total sample count
            if per_module_gradients[self.lora_module_names[0]]:  # Check if we have at least one gradient
                batch_size = per_module_gradients[self.lora_module_names[0]][-1].shape[0]
                total_samples += batch_size

        print(f"Collected gradients from {len(per_module_gradients[self.lora_module_names[0]])} batches, total samples: {total_samples}")

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
            num_layers = len(self.lora_module_names)
            num_samples = next(iter(concatenated_gradients.values())).shape[0]
            per_sample_shape = next(iter(gradient_shapes.values()))

            # Initialize the final tensor
            device = next(iter(concatenated_gradients.values())).device
            gradients = torch.zeros((num_layers, num_samples) + per_sample_shape, device=device)

            # Fill the tensor with gradients
            for name, tensor in concatenated_gradients.items():
                module_idx = self.lora_module_to_idx[name]
                gradients[module_idx] = tensor

        else:
            # Gradient shapes differ, keep as dictionary with module indices
            gradients = {self.lora_module_to_idx[name]: tensor for name, tensor in concatenated_gradients.items()}

        # Store or return gradients based on dataset type
        if dataset_type == "train":
            self.train_gradients = gradients
            return self.train_gradients
        else:
            self.test_gradients = gradients
            return self.test_gradients

    def _precondition_gradients_raw(self, gradients, module_name):
        """
        Precondition gradients using raw Hessian (gradient covariance) approximation.

        Args:
            gradients: Gradient tensor
            module_name: Name of the module

        Returns:
            Preconditioned gradients
        """
        # Ensure covariance inverse is computed
        if not hasattr(self, 'covariance_inverse'):
            self._compute_grad_covariance_inverse()

        # Get inverse covariance
        inverse_cov = self.cov_inv[module_name]["grad"]

        # Precondition gradients: G * H^-1
        precond_grads = torch.matmul(gradients, inverse_cov)

        return precond_grads

    def _precondition_gradients_kfac(self, gradients, module_name):
        """
        Precondition gradients using K-FAC approximation.

        Args:
            gradients: Gradient tensor
            module_name: Name of the module

        Returns:
            Preconditioned gradients
        """
        # Ensure eigendecomposition is already computed
        if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
            raise ValueError("Eigendecomposition must be computed before using K-FAC")

        # Get eigenvectors and eigenvalues
        fwd_eigvec = self.eigenvectors[module_name]['forward']
        bwd_eigvec = self.eigenvectors[module_name]['backward']
        fwd_eigval = self.eigenvalues[module_name]['forward']
        bwd_eigval = self.eigenvalues[module_name]['backward']

        # Create full eigenvalue matrix (outer product)
        full_eigval = torch.outer(bwd_eigval, fwd_eigval)

        # Add damping
        if self.damping is None:
            self.damping = 0.1 * torch.mean(full_eigval)
        full_eigval += self.damping

        original_shape = gradients.shape
        # Reshape gradients to matrix form
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

    def _precondition_gradients_ekfac(self, gradients, module_name):
        """
        Precondition gradients using EK-FAC approximation.

        Args:
            gradients: Gradient tensor
            module_name: Name of the module

        Returns:
            Preconditioned gradients
        """
        # Ensure EK-FAC eigenvalues are already computed
        if not hasattr(self, 'ekfac_eigenvalues') or not self.ekfac_eigenvalues:
            raise ValueError("EK-FAC eigenvalues must be computed before using EK-FAC")

        # Get eigenvectors and EK-FAC eigenvalues
        fwd_eigvec = self.eigenvectors[module_name]['forward']
        bwd_eigvec = self.eigenvectors[module_name]['backward']
        ekfac_eigval = self.ekfac_eigenvalues[module_name]

        # Add damping
        ekfac_eigval_damped = ekfac_eigval + self.damping

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

            # Precondition with inverse EK-FAC eigenvalues
            precond_rotated = rotated_grad / ekfac_eigval_damped

            # Rotate back
            precond_grad = torch.matmul(torch.matmul(bwd_eigvec, precond_rotated), fwd_eigvec.t())

            precond_grads.append(precond_grad.flatten())

        # Stack and reshape back to original shape
        return torch.stack(precond_grads).reshape(original_shape)

    def attribute(
            self,
            test_dataloader: torch.utils.data.DataLoader,
            train_dataloader: Optional[torch.utils.data.DataLoader] = None
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Attributing the test set with respect to the training set.

        Args:
            test_dataloader (torch.utils.data.DataLoader): _description_
            train_dataloader (Optional[torch.utils.data.DataLoader], optional): _description_. Defaults to None.

        Returns:
            If profile=False:
                torch.Tensor: The influence scores
            If profile=True:
                Tuple[torch.Tensor, Dict]: The influence scores and profiling statistics
        """
        if self.train_gradients is None:
            raise ValueError("No training gradients collected. Call collect_gradients first.")


        test_gradients = self.collect_gradients(test_dataloader, dataset_type="test")

        print(f"Computing influence scores...")

        # Both gradients are tensors with shape (num_layers, num_samples, *per_sample_gradient_shape)
        num_train_samples = self.train_gradients.shape[1]
        num_test_samples = test_gradients.shape[1]

        # Initialize influence matrix
        IF_score = torch.zeros((num_train_samples, num_test_samples), device=self.train_gradients.device)

        if self.profile:
            torch.cuda.synchronize()
            start_time = time.time()

        # Compute influence scores based on hessian approximation
        if self.hessian == "none":
            for module_name in self.lora_module_names:
                layer_idx = self.lora_module_to_idx[module_name]
                test_layer_grads = test_gradients[layer_idx]
                train_layer_grads = self.train_gradients[layer_idx]
                layer_influence = torch.matmul(train_layer_grads, test_layer_grads.t())
                IF_score += layer_influence

        elif self.hessian == "raw":
            if not hasattr(self, 'covariance_inverse'):
                print("Computing Fisher information matrix inverse...")
                self._compute_grad_covariance_inverse()

            for module_name in self.lora_module_names:
                layer_idx = self.lora_module_to_idx[module_name]

                # Skip if we don't have covariance for this module
                if module_name not in self.cov_inv:
                    continue

                test_layer_grads = test_gradients[layer_idx]
                train_layer_grads = self.train_gradients[layer_idx]
                precond_train_grads = self._precondition_gradients_raw(train_layer_grads, module_name)

                layer_influence = torch.matmul(precond_train_grads, test_layer_grads.t())
                IF_score += layer_influence

        elif self.hessian == "kfac":
            if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
                print("Computing eigendecomposition for K-FAC...")
                self._compute_eigendecomposition()

            for module_name in self.lora_module_names:
                layer_idx = self.lora_module_to_idx[module_name]

                # Skip if we don't have eigendecomposition for this module
                if module_name not in self.eigenvectors:
                    continue

                # Extract gradients for this layer
                test_layer_grads = test_gradients[layer_idx]
                train_layer_grads = self.train_gradients[layer_idx].view(num_train_samples, self.rank, self.rank)

                precond_train_grads = self._precondition_gradients_kfac(train_layer_grads, module_name)

                layer_influence = torch.matmul(precond_train_grads, test_layer_grads.t())
                IF_score += layer_influence

        elif self.hessian == "ekfac":
            if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
                print("Computing eigendecomposition for EK-FAC...")
                self._compute_eigendecomposition()

            if not hasattr(self, 'ekfac_eigenvalues') or not self.ekfac_eigenvalues:
                raise ValueError("EK-FAC eigenvalues must be computed before using EK-FAC. Run _compute_ekfac_eigenvalues first.")

            for module_name in self.lora_module_names:
                layer_idx = self.lora_module_to_idx[module_name]

                # Skip if we don't have EK-FAC data for this module
                if module_name not in self.ekfac_eigenvalues:
                    continue

                # Extract gradients for this layer
                test_layer_grads = test_gradients[layer_idx]
                train_layer_grads = self.train_gradients[layer_idx]

                precond_train_grads = self._precondition_gradients_ekfac(train_layer_grads, module_name)

                layer_influence = torch.matmul(precond_train_grads, test_layer_grads.t())
                IF_score += layer_influence

        else:
            raise ValueError(f"Unsupported hessian approximation: {self.hessian}")

        if self.profile:
            torch.cuda.synchronize()
            self.profiling_stats['precondition'] += time.time() - start_time

        return (IF_score, self.profiling_stats) if self.profile else IF_score