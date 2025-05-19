"""
Concrete implementation of the Influence Function Attributor.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union, Tuple, Any
import time
import gc

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

import torch
from tqdm import tqdm

from .base import BaseAttributor, HessianOptions, ProfilingStats
from ..utils.common import stable_inverse

import logging
logger = logging.getLogger(__name__)

class IFAttributor(BaseAttributor):
    """
    Influence function calculator using hooks for efficient gradient projection.
    Works with standard PyTorch layers with support for different offloading strategies.
    """

    def compute_preconditioners(self, damping: Optional[float] = None) -> List[torch.Tensor]:
        """
        Compute preconditioners (inverse Hessian) from gradients.
        Supports batch_size > 1 by properly aggregating gradients before computing the preconditioner.

        Args:
            damping: (Adaptive) damping factor for Hessian inverse

        Returns:
            List of preconditioners for each layer
        """
        logger.info(f"Computing preconditioners with hessian type: {self.hessian}")

        # Use instance damping if not provided
        if damping is None:
            damping = self.damping

        # Validation code
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found. Call cache_gradients first.")

        # If hessian type is "none", no preconditioners needed
        if self.hessian == "none":
            logger.info("Hessian type is 'none', skipping preconditioner computation")
            # Store None preconditioners in the strategy
            for layer_idx in range(len(self.layer_names)):
                self.strategy.store_preconditioner(layer_idx, None)
            return [None] * len(self.layer_names)

        # Calculate total samples across all batches
        total_samples = self.metadata.get_total_samples()
        logger.info(f"Total samples across all batches: {total_samples}")

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Initialize Hessian accumulators for all layers
        hessian_accumulators = [None] * len(self.layer_names)
        sample_counts = [0] * len(self.layer_names)

        # Process batches - if disk offload, use dataloader, otherwise process directly
        dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=4,  # Process multiple files at a time
            num_workers=4,
            pin_memory=True
        )

        if dataloader:
            # Process batches using the dataloader (disk offload)
            for batch_indices, batch_dicts in tqdm(dataloader, desc="Processing batches for preconditioners"):
                # Process each layer
                for layer_idx in range(len(self.layer_names)):
                    # Collect all gradients for this layer from the batch
                    layer_gradients = []

                    for i, batch_dict in enumerate(batch_dicts):
                        # Check if this layer exists in the batch and has data
                        if layer_idx not in batch_dict or batch_dict[layer_idx].numel() == 0:
                            continue

                        grad = batch_dict[layer_idx]

                        layer_gradients.append(grad)

                    # If no gradients for this layer, skip
                    if not layer_gradients:
                        continue

                    # Aggregate all gradients into one tensor
                    aggregated_gradients = torch.cat(layer_gradients, dim=0)

                    # Move aggregated gradients to device
                    aggregated_gradients = self.strategy.move_to_device(aggregated_gradients)

                    # Update sample count
                    sample_counts[layer_idx] += aggregated_gradients.shape[0]

                    # Compute Hessian contribution from the aggregated batch
                    batch_hessian = torch.matmul(aggregated_gradients.t(), aggregated_gradients)

                    # Update the Hessian accumulator
                    if hessian_accumulators[layer_idx] is None:
                        hessian_accumulators[layer_idx] = batch_hessian
                    else:
                        hessian_accumulators[layer_idx] += batch_hessian

                    # Clean up memory
                    del aggregated_gradients, batch_hessian, layer_gradients
                    torch.cuda.empty_cache()

                # Clean up memory after processing all layers for this batch
                gc.collect()
                torch.cuda.empty_cache()
        else:
            # Get all batch indices from metadata
            batch_indices = sorted(self.metadata.batch_info.keys())
            chunk_size = self.default_chunk_size

            # Process in chunks to manage memory
            for chunk_start in range(0, len(batch_indices), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(batch_indices))
                chunk_batch_indices = batch_indices[chunk_start:chunk_end]

                # Process each layer for all batches in the chunk
                for layer_idx in range(len(self.layer_names)):
                    gradients_list = []

                    # Collect gradients for this layer from all batches in the chunk
                    for batch_idx in chunk_batch_indices:
                        batch_grads = self.strategy.retrieve_gradients(batch_idx, is_test=False)
                        if layer_idx < len(batch_grads) and batch_grads[layer_idx].numel() > 0:
                            gradients_list.append(batch_grads[layer_idx])

                    if not gradients_list:
                        continue

                    # Process in smaller batches if many gradients
                    grad_batch_size = 100  # Adjust based on memory constraints
                    for i in range(0, len(gradients_list), grad_batch_size):
                        batch_grads = gradients_list[i:i+grad_batch_size]

                        # Combine gradients
                        combined_gradients = torch.cat(batch_grads, dim=0)

                        # Compute Hessian contribution
                        batch_hessian = torch.matmul(combined_gradients.t(), combined_gradients)

                        # Update accumulator
                        if hessian_accumulators[layer_idx] is None:
                            hessian_accumulators[layer_idx] = batch_hessian
                        else:
                            hessian_accumulators[layer_idx] += batch_hessian

                        # Update sample count
                        sample_counts[layer_idx] += combined_gradients.shape[0]

                        # Clean up
                        del combined_gradients, batch_hessian
                        torch.cuda.empty_cache()

                # Clean up memory after processing the chunk
                gc.collect()
                torch.cuda.empty_cache()

        # Compute preconditioners from accumulated Hessians
        preconditioners = [None] * len(self.layer_names)

        # Process each layer's accumulated Hessian
        for layer_idx in tqdm(range(len(self.layer_names)), desc="Computing preconditioners"):
            hessian_accumulator = hessian_accumulators[layer_idx]
            sample_count = sample_counts[layer_idx]

            # If we have accumulated Hessian, compute preconditioner
            if hessian_accumulator is not None and sample_count > 0:
                # Normalize by total number of samples
                hessian = hessian_accumulator / sample_count

                # Compute inverse based on Hessian type
                if self.hessian == "raw":
                    precond = stable_inverse(hessian, damping=damping)
                    preconditioners[layer_idx] = precond

                    # Store in strategy
                    self.strategy.store_preconditioner(layer_idx, precond)

                    # Clean up
                    del precond

                elif self.hessian in ["kfac", "ekfac"]:
                    # Store Hessian itself for KFAC-type preconditioners
                    preconditioners[layer_idx] = hessian

                    # Store in strategy
                    self.strategy.store_preconditioner(layer_idx, hessian)

                # Clean up
                del hessian_accumulator, hessian
                torch.cuda.empty_cache()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

        # Wait for async operations to complete
        self.strategy.wait_for_async_operations()

        if self.profile and self.profiling_stats:
            return (preconditioners, self.profiling_stats)
        else:
            return preconditioners

    def compute_ifvp(
        self,
        batch_range: Optional[Tuple[int, int]] = None
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Compute inverse-Hessian-vector products (IFVP) from gradients and preconditioners.
        Optimized for minimal memory usage with batch processing.

        Args:
            batch_range: Optional tuple of (start_batch, end_batch) to process only a subset of batches

        Returns:
            Dictionary mapping batch indices to lists of tensors (one tensor per layer)
        """
        logger.info("Computing inverse-Hessian-vector products (IFVP)")

        # Load batch information if needed
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found. Call cache_gradients first.")

        # Process batch range
        batch_msg = ""
        if batch_range is not None:
            start_batch, end_batch = batch_range
            batch_msg = f" (processing batches {start_batch} to {end_batch-1})"
        logger.info(f"Computing IFVP with offload strategy: {self.offload}{batch_msg}")

        # Return raw gradients if Hessian type is "none"
        if self.hessian == "none":
            logger.info("Using raw gradients as IFVP since hessian type is 'none'")

            # Get all batch indices from metadata
            if batch_range is not None:
                start_batch, end_batch = batch_range
                batch_indices = [idx for idx in self.metadata.batch_info.keys()
                              if start_batch <= idx < end_batch]
            else:
                batch_indices = sorted(self.metadata.batch_info.keys())

            # Copy gradients to IFVP storage
            for batch_idx in tqdm(batch_indices, desc="Copying gradients to IFVP"):
                gradients = self.strategy.retrieve_gradients(batch_idx, is_test=False)
                self.strategy.store_ifvp(batch_idx, gradients)

            # Create minimal result dictionary
            result_dict = {batch_idx: None for batch_idx in batch_indices}
            return result_dict

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Get all batch indices from metadata
        if batch_range is not None:
            start_batch, end_batch = batch_range
            batch_indices = [idx for idx in self.metadata.batch_info.keys()
                          if start_batch <= idx < end_batch]
        else:
            batch_indices = sorted(self.metadata.batch_info.keys())

        # Load all preconditioners at once to avoid repeated loading
        logger.info("Loading preconditioners for all layers")
        preconditioners = []
        for layer_idx in range(len(self.layer_names)):
            precond = self.strategy.retrieve_preconditioner(layer_idx)
            preconditioners.append(precond)  # Will be None if not available

        # Log how many preconditioners were loaded
        valid_preconditioners = sum(1 for p in preconditioners if p is not None)
        logger.info(f"Loaded {valid_preconditioners} preconditioners out of {len(self.layer_names)} layers")

        # Process batches - if disk offload, use dataloader, otherwise process directly
        dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=4,
            num_workers=32,
            pin_memory=True,
            batch_range=batch_range
        )

        # Initialize result dictionary
        result_dict = {}

        if dataloader:
            # Process batches using the dataloader (disk offload)
            for batch_indices, batch_grad_dicts in tqdm(dataloader, desc="Computing IFVP"):
                # First, organize all the batch info for tracking
                batch_ifvp_dicts = [{} for _ in range(len(batch_indices))]

                # Process each layer
                for layer_idx in range(len(self.layer_names)):
                    # Get preconditioner for this layer from our preloaded list
                    precond = preconditioners[layer_idx]

                    # Skip if preconditioner is None
                    if precond is None:
                        continue

                    # Collect all gradients for this layer from all batches
                    batch_gradients = []
                    batch_dict_indices = []  # To track which position in the batch each gradient came from

                    # Get info for each batch
                    for batch_dict_idx, batch_grad_dict in enumerate(batch_grad_dicts):
                        # Skip if layer not in gradient dictionary or empty
                        if layer_idx not in batch_grad_dict or batch_grad_dict[layer_idx].numel() == 0:
                            batch_ifvp_dicts[batch_dict_idx][layer_idx] = torch.tensor([])
                            continue

                        grad = batch_grad_dict[layer_idx]

                        # Save the gradient along with its batch info
                        batch_gradients.append(grad)
                        batch_dict_indices.append(batch_dict_idx)

                    # If no valid gradients for this layer, continue to next layer
                    if not batch_gradients:
                        continue

                    # Move gradients to device and concatenate
                    device_gradients = [self.strategy.move_to_device(grad) for grad in batch_gradients]

                    # Store original shapes for splitting results later
                    original_shapes = [grad.shape[0] for grad in device_gradients]

                    # Concatenate all gradients for efficient processing
                    concatenated_gradients = torch.cat(device_gradients, dim=0)

                    # Move preconditioner to device and ensure types match
                    device_precond = self.strategy.move_to_device(precond)
                    device_precond = device_precond.to(dtype=concatenated_gradients.dtype)

                    # Compute IFVP for all gradients at once
                    result = torch.matmul(device_precond, concatenated_gradients.t()).t()

                    # Split result back to individual batch results
                    split_results = torch.split(result, original_shapes)

                    # Distribute results back to their respective batch dictionaries
                    for i, (split_result, dict_idx) in enumerate(zip(split_results, batch_dict_indices)):
                        batch_ifvp_dicts[dict_idx][layer_idx] = self.strategy.move_from_device(split_result)

                    # Clean up memory
                    del device_gradients, concatenated_gradients, result, split_results, device_precond
                    torch.cuda.empty_cache()

                # Save all computed IFVP results
                for i, (batch_idx, batch_ifvp_dict) in enumerate(zip(batch_indices, batch_ifvp_dicts)):
                    # Convert to list for consistent interface
                    batch_ifvp_list = []
                    for layer_idx in range(len(self.layer_names)):
                        if layer_idx in batch_ifvp_dict:
                            batch_ifvp_list.append(batch_ifvp_dict[layer_idx])
                        else:
                            batch_ifvp_list.append(torch.tensor([]))

                    # Store in strategy
                    self.strategy.store_ifvp(batch_idx, batch_ifvp_list)

                    # Add to result dictionary
                    result_dict[batch_idx] = batch_ifvp_list

                # Clean up memory after processing all layers for these batches
                gc.collect()
                torch.cuda.empty_cache()
        else:
            # Process batches directly (memory or CPU offload)
            for batch_idx in tqdm(batch_indices, desc="Computing IFVP"):
                # Initialize IFVP list for this batch
                batch_ifvp = [torch.tensor([]) for _ in range(len(self.layer_names))]

                # Get gradients for this batch
                batch_grads = self.strategy.retrieve_gradients(batch_idx, is_test=False)

                # Process each layer
                for layer_idx in range(len(self.layer_names)):
                    # Skip if gradient is empty
                    if layer_idx >= len(batch_grads) or batch_grads[layer_idx].numel() == 0:
                        continue

                    # Get preconditioner for this layer from preloaded list
                    precond = preconditioners[layer_idx]

                    # Skip if preconditioner is None
                    if precond is None:
                        continue

                    # Get gradient and ensure it's on the device
                    grad = self.strategy.move_to_device(batch_grads[layer_idx])

                    # Move preconditioner to device and ensure types match
                    device_precond = self.strategy.move_to_device(precond)
                    device_precond = device_precond.to(dtype=grad.dtype)

                    # Compute IFVP
                    ifvp = torch.matmul(device_precond, grad.t()).t()

                    # Store result
                    batch_ifvp[layer_idx] = self.strategy.move_from_device(ifvp)

                    # Clean up
                    del grad, ifvp, device_precond
                    torch.cuda.empty_cache()

                # Store in strategy
                self.strategy.store_ifvp(batch_idx, batch_ifvp)

                # Add to result dictionary
                result_dict[batch_idx] = batch_ifvp

        # Clean up preloaded preconditioners
        del preconditioners
        gc.collect()
        torch.cuda.empty_cache()

        # Wait for async operations to complete
        self.strategy.wait_for_async_operations()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time
            return (result_dict, self.profiling_stats)
        else:
            return result_dict

    def compute_self_influence(
        self,
        batch_range: Optional[Tuple[int, int]] = None,
        precondition: bool = True,
        damping: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute self-influence scores for training examples.
        This is useful for uncertainty estimation.

        Args:
            batch_range: Optional tuple of (start_batch, end_batch) to process only a subset of batches
            precondition: Whether to use preconditioned gradients
            damping: Optional damping parameter (uses self.damping if None)

        Returns:
            Tensor containing self-influence scores for all examples
        """
        logger.info("Computing self-influence scores")

        # Load batch information if needed
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found. Call cache_gradients first.")

        # Get batch mapping and determine total samples
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        total_samples = self.metadata.get_total_samples()

        # Initialize result tensor
        self_influence = torch.zeros(total_samples, device="cpu")

        # Filter batch indices based on batch_range
        if batch_range is not None:
            start_batch, end_batch = batch_range
            batch_indices = [idx for idx in self.metadata.batch_info.keys()
                           if start_batch <= idx < end_batch]
        else:
            batch_indices = sorted(self.metadata.batch_info.keys())

        # Process batches - if disk offload, use dataloader, otherwise process directly
        dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=8,
            num_workers=32,
            pin_memory=True,
            batch_range=batch_range
        )

        if dataloader:
            # Process batches using the dataloader (disk offload)
            for batch_indices, batch_grad_dicts in tqdm(dataloader, desc="Computing self-influence"):
                # Process each individual batch
                for batch_idx, batch_grad_dict in zip(batch_indices, batch_grad_dicts):
                    # Get sample range for this batch
                    if batch_idx not in batch_to_sample_mapping:
                        logger.warning(f"Batch {batch_idx} not found in mapping, skipping")
                        continue

                    sample_start, sample_end = batch_to_sample_mapping[batch_idx]
                    num_samples = sample_end - sample_start

                    # Process each layer
                    for layer_idx in range(len(self.layer_names)):
                        if layer_idx not in batch_grad_dict or batch_grad_dict[layer_idx].numel() == 0:
                            continue

                        # Get gradient and move to device
                        grad = self.strategy.move_to_device(batch_grad_dict[layer_idx])

                        if precondition:
                            # Get preconditioner
                            precond = self.strategy.retrieve_preconditioner(layer_idx)

                            if precond is not None:
                                # Calculate preconditioned self-influence
                                precond_grad = torch.matmul(precond, grad.t()).t()
                                layer_self_influence = torch.sum(grad * precond_grad, dim=1)

                                # Clean up
                                del precond_grad, precond
                            else:
                                # Fall back to raw self-influence
                                layer_self_influence = torch.sum(grad * grad, dim=1)
                        else:
                            # Calculate raw self-influence
                            layer_self_influence = torch.sum(grad * grad, dim=1)

                        # Make sure the shape is correct
                        if layer_self_influence.shape[0] != num_samples:
                            logger.warning(f"Influence shape {layer_self_influence.shape[0]} doesn't match expected sample count {num_samples}")
                            # Try to reshape if possible
                            if layer_self_influence.numel() == num_samples:
                                layer_self_influence = layer_self_influence.reshape(num_samples)
                            else:
                                # Skip this layer if shapes don't match and can't be fixed
                                continue

                        # Add to total self-influence
                        self_influence[sample_start:sample_end] += layer_self_influence.cpu()

                        # Clean up
                        del grad, layer_self_influence
                        torch.cuda.empty_cache()

                # Clean up after processing each DataLoader batch
                gc.collect()
                torch.cuda.empty_cache()
        else:
            # Process batches directly (memory or CPU offload)
            for batch_idx in tqdm(batch_indices, desc="Computing self-influence"):
                # Get sample range for this batch
                if batch_idx not in batch_to_sample_mapping:
                    logger.warning(f"Batch {batch_idx} not found in mapping, skipping")
                    continue

                sample_start, sample_end = batch_to_sample_mapping[batch_idx]
                num_samples = sample_end - sample_start

                # Get gradients for this batch
                batch_grads = self.strategy.retrieve_gradients(batch_idx, is_test=False)

                # Process each layer
                for layer_idx in range(len(self.layer_names)):
                    if layer_idx >= len(batch_grads) or batch_grads[layer_idx].numel() == 0:
                        continue

                    # Get gradient
                    grad = batch_grads[layer_idx]

                    if precondition:
                        # Get preconditioner
                        precond = self.strategy.retrieve_preconditioner(layer_idx)

                        if precond is not None:
                            # Calculate preconditioned self-influence
                            precond_grad = torch.matmul(precond, grad.t()).t()
                            layer_self_influence = torch.sum(grad * precond_grad, dim=1)

                            # Clean up
                            del precond_grad, precond
                        else:
                            # Fall back to raw self-influence
                            layer_self_influence = torch.sum(grad * grad, dim=1)
                    else:
                        # Calculate raw self-influence
                        layer_self_influence = torch.sum(grad * grad, dim=1)

                    # Make sure the shape is correct
                    if layer_self_influence.shape[0] != num_samples:
                        logger.warning(f"Influence shape {layer_self_influence.shape[0]} doesn't match expected sample count {num_samples}")
                        # Try to reshape if possible
                        if layer_self_influence.numel() == num_samples:
                            layer_self_influence = layer_self_influence.reshape(num_samples)
                        else:
                            # Skip this layer if shapes don't match and can't be fixed
                            continue

                    # Add to total self-influence
                    self_influence[sample_start:sample_end] += layer_self_influence.cpu()

                    # Clean up
                    del grad, layer_self_influence
                    torch.cuda.empty_cache()

                # Clean up after processing each batch
                gc.collect()
                torch.cuda.empty_cache()

        return self_influence

    def attribute(
        self,
        test_dataloader: 'DataLoader',
        train_dataloader: Optional['DataLoader'] = None,
        use_cached_ifvp: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ProfilingStats]]:
        """
        Attribute influence of training examples on test examples.
        Optimized to process multiple batches simultaneously with proper aggregation
        and reduced device transfers.

        Args:
            test_dataloader: DataLoader for test data
            train_dataloader: Optional DataLoader for training data if not cached
            use_cached_ifvp: Whether to use cached IFVP (True) or recompute from cached gradients (False)

        Returns:
            Tensor of influence scores (and profiling stats if profile=True)
        """
        # Load batch information if needed
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            if train_dataloader is None:
                raise ValueError("No batch information found and no training dataloader provided.")

            # Cache gradients if needed
            logger.info("No batch metadata found. Caching gradients from provided dataloader...")
            self.cache_gradients(train_dataloader)

        # Set sparsifiers in the hook manager
        if self.sparsifiers is None and self.projectors is None:
            self._setup_compressors(test_dataloader)

        # Get or compute IFVP
        if use_cached_ifvp and self.strategy.has_ifvp():
            logger.info("Using cached IFVP")
        else:
            logger.info("Computing IFVP")
            self.compute_ifvp()

        # Compute test gradients
        test_grads_dict, _ = self._compute_gradients(
            test_dataloader,
            is_test=True,
            batch_range=(0, len(test_dataloader))
        )

        torch.cuda.empty_cache()

        # Calculate total training samples and map batches to sample indices
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        total_train_samples = self.metadata.get_total_samples()

        # Initialize influence scores in memory
        num_test = len(test_dataloader.dataset)
        IF_score = torch.zeros(total_train_samples, num_test, device="cpu")

        # Map test batch indices to sample ranges and organize test gradients by layer
        test_batch_indices = {}
        current_index = 0

        # Organize test gradients by layer for efficiency
        test_grads_by_layer = {}

        # First pass: determine sample indices for each test batch
        for test_batch_idx in sorted(test_grads_dict.keys()):
            # Find first non-empty layer to get batch size
            batch_size = 0
            for layer_grads in test_grads_dict[test_batch_idx]:
                if layer_grads.numel() > 0:
                    batch_size = layer_grads.shape[0]
                    break

            if batch_size == 0:
                logger.warning(f"Could not determine batch size for test batch {test_batch_idx}")
                continue

            test_batch_indices[test_batch_idx] = (current_index, current_index + batch_size)
            current_index += batch_size

        # Second pass: organize test gradients by layer
        for test_batch_idx, test_grads in test_grads_dict.items():
            if test_batch_idx not in test_batch_indices:
                continue

            col_st, col_ed = test_batch_indices[test_batch_idx]

            for layer_idx, grad in enumerate(test_grads):
                if grad.numel() == 0:
                    continue

                # Store gradient with its column range
                if layer_idx not in test_grads_by_layer:
                    test_grads_by_layer[layer_idx] = []

                test_grads_by_layer[layer_idx].append((grad, col_st, col_ed))

        # Remove hooks after collecting test gradients
        if self.hook_manager:
            self.hook_manager.remove_hooks()
            self.hook_manager = None

        # Process batches - if disk offload, use dataloader, otherwise process directly
        train_batch_indices = sorted(self.metadata.batch_info.keys())
        dataloader = self.strategy.create_gradient_dataloader(
            data_type="ifvp",
            batch_size=8,  # Process multiple files at a time
            num_workers=32,
            pin_memory=True
        )

        if dataloader:
            # Process batches using the dataloader (disk offload)
            for train_batch_indices, train_batch_dicts in tqdm(dataloader, desc="Processing train batches"):
                # Process each layer
                for layer_idx in range(len(self.layer_names)):
                    # Skip if no test gradients for this layer
                    if layer_idx not in test_grads_by_layer:
                        continue

                    # Get test gradients for this layer
                    layer_test_grads = test_grads_by_layer[layer_idx]

                    # Collect IFVP data for all train batches for this layer
                    layer_train_ifvps = []
                    layer_train_row_ranges = []

                    # Get IFVP for all train batches in this loader batch
                    for train_batch_idx, train_batch_dict in zip(train_batch_indices, train_batch_dicts):
                        # Skip if layer not in train batch or empty
                        if layer_idx not in train_batch_dict or train_batch_dict[layer_idx].numel() == 0:
                            continue

                        # Get row indices for this train batch
                        if train_batch_idx not in batch_to_sample_mapping:
                            logger.warning(f"Train batch {train_batch_idx} not found in mapping, skipping")
                            continue

                        row_st, row_ed = batch_to_sample_mapping[train_batch_idx]
                        train_samples = row_ed - row_st

                        train_ifvp = train_batch_dict[layer_idx]
                        layer_train_ifvps.append(train_ifvp)
                        layer_train_row_ranges.append((row_st, row_ed))

                    # Skip if no train data for this layer
                    if not layer_train_ifvps:
                        continue

                    # Move all train IFVP data to device
                    layer_train_ifvps_gpu = [self.strategy.move_to_device(ifvp) for ifvp in layer_train_ifvps]

                    # Process all test gradients for this layer
                    for test_grad, col_st, col_ed in layer_test_grads:
                        test_grad_gpu = self.strategy.move_to_device(test_grad)

                        for ifvp_gpu, (row_st, row_ed) in zip(layer_train_ifvps_gpu, layer_train_row_ranges):
                            layer_influence = torch.matmul(ifvp_gpu, test_grad_gpu.t())
                            IF_score[row_st:row_ed, col_st:col_ed] += layer_influence.cpu()

                            # Clean up
                            del layer_influence

                        # Clean up test gradient
                        del test_grad_gpu
                        torch.cuda.empty_cache()

                    # Clean up train IFVP data
                    del layer_train_ifvps_gpu
                    torch.cuda.empty_cache()

                # Clean up after processing this layer
                gc.collect()
                torch.cuda.empty_cache()
        else:
            # Process batches directly (memory or CPU offload)
            for train_batch_idx in tqdm(train_batch_indices, desc="Processing train batches"):
                # Get row indices for this train batch
                if train_batch_idx not in batch_to_sample_mapping:
                    logger.warning(f"Train batch {train_batch_idx} not found in mapping, skipping")
                    continue

                row_st, row_ed = batch_to_sample_mapping[train_batch_idx]

                # Get IFVP for this batch
                train_ifvp = self.strategy.retrieve_ifvp(train_batch_idx)

                # Process each layer
                for layer_idx in range(len(self.layer_names)):
                    # Skip if no test gradients for this layer
                    if layer_idx not in test_grads_by_layer:
                        continue

                    # Skip if IFVP is empty for this layer
                    if layer_idx >= len(train_ifvp) or train_ifvp[layer_idx].numel() == 0:
                        continue

                    layer_ifvp = train_ifvp[layer_idx]
                    layer_ifvp = self.strategy.move_to_device(layer_ifvp)

                    # Process all test gradients for this layer
                    for test_grad, col_st, col_ed in test_grads_by_layer[layer_idx]:
                        test_grad = self.strategy.move_to_device(test_grad)
                        layer_influence = torch.matmul(layer_ifvp, test_grad.t())
                        IF_score[row_st:row_ed, col_st:col_ed] += layer_influence.cpu()

                        # Clean up
                        del layer_influence, test_grad
                        torch.cuda.empty_cache()

                    # Clean up
                    del layer_ifvp
                    torch.cuda.empty_cache()

                # Clean up
                del train_ifvp
                gc.collect()
                torch.cuda.empty_cache()

        # Return result
        if self.profile and self.profiling_stats:
            return (IF_score, self.profiling_stats)
        else:
            return IF_score