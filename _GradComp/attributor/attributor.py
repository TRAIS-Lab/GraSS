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
    ) -> torch.Tensor:
        """
        Compute self-influence scores for training examples by using
        the dot product of gradients and their IFVPs.

        Optimized by:
        1. Processing in dataloader chunks
        2. Processing each layer separately
        3. Concatenating all batches in a chunk for each layer

        Args:
            batch_range: Optional tuple of (start_batch, end_batch) to process only a subset of batches

        Returns:
            Tensor containing self-influence scores for all examples
        """
        logger.info("Computing self-influence scores using pre-computed IFVP")

        # Load batch information if needed
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found. Call cache_gradients first.")

        # Make sure IFVP is computed
        if not self.strategy.has_ifvp():
            logger.info("IFVP not found, computing it now...")
            self.compute_ifvp(batch_range=batch_range)

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

        # Verify batches are continuous
        if len(batch_indices) > 1:
            for i in range(len(batch_indices) - 1):
                if batch_indices[i] + 1 != batch_indices[i+1]:
                    raise ValueError(f"Batch indices must be continuous. Found gap between "
                                f"batch {batch_indices[i]} and {batch_indices[i+1]}.")

        # Create dataloaders with larger batch size for better efficiency
        dataloader_batch_size = 128  # Process larger chunks
        grad_dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=dataloader_batch_size,
            num_workers=4,  # More workers for parallel I/O
            pin_memory=True,
            batch_range=batch_range
        )

        ifvp_dataloader = self.strategy.create_gradient_dataloader(
            data_type="ifvp",
            batch_size=dataloader_batch_size,
            num_workers=4,
            pin_memory=True,
            batch_range=batch_range
        )

        # Process batches with dataloader
        if grad_dataloader and ifvp_dataloader:
            # First loop: Process chunks from the dataloader
            for (grad_batch_indices, grad_batch_dicts), (ifvp_batch_indices, ifvp_batch_dicts) in tqdm(
                zip(grad_dataloader, ifvp_dataloader),
                desc="Processing batch chunks",
                total=len(grad_dataloader)
            ):
                # Verify batch indices match
                if grad_batch_indices != ifvp_batch_indices:
                    raise ValueError(f"Batch indices mismatch: {grad_batch_indices} vs {ifvp_batch_indices}")

                # Collect sample information for this chunk
                sample_info = []  # List of (batch_idx, batch_pos, sample_start, num_samples)

                for batch_pos, batch_idx in enumerate(grad_batch_indices):
                    if batch_idx not in batch_to_sample_mapping:
                        continue

                    sample_start, sample_end = batch_to_sample_mapping[batch_idx]
                    num_samples = sample_end - sample_start

                    if num_samples <= 0:
                        continue

                    sample_info.append((batch_idx, batch_pos, sample_start, num_samples))

                # Skip if no valid batches in chunk
                if not sample_info:
                    continue

                # Second loop: Process each layer separately for the entire chunk
                for layer_idx in range(len(self.layer_names)):
                    # Collect gradients and IFVPs for this layer across all batches in the chunk
                    layer_grads = []
                    layer_ifvps = []
                    layer_sample_starts = []
                    layer_sample_counts = []

                    # Gather data for this layer from all batches in the chunk
                    for batch_idx, batch_pos, sample_start, num_samples in sample_info:
                        # Skip if layer data is missing in this batch
                        if (layer_idx not in grad_batch_dicts[batch_pos] or
                            grad_batch_dicts[batch_pos][layer_idx].numel() == 0 or
                            layer_idx not in ifvp_batch_dicts[batch_pos] or
                            ifvp_batch_dicts[batch_pos][layer_idx].numel() == 0):
                            continue

                        # Get data for this layer/batch
                        grad = grad_batch_dicts[batch_pos][layer_idx]
                        ifvp = ifvp_batch_dicts[batch_pos][layer_idx]

                        # Validate shapes
                        if grad.shape[0] != num_samples or ifvp.shape[0] != num_samples:
                            raise ValueError(f"Shape mismatch for batch {batch_idx}, layer {layer_idx}. "
                                            f"Expected {num_samples}, got {grad.shape[0]}/{ifvp.shape[0]}")

                        # Add to collection
                        layer_grads.append(grad)
                        layer_ifvps.append(ifvp)
                        layer_sample_starts.append(sample_start)
                        layer_sample_counts.append(num_samples)

                    # Skip layer if no valid data in any batch
                    if not layer_grads:
                        continue

                    # Concatenate all batch data for this layer
                    try:
                        cat_grads = torch.cat(layer_grads, dim=0)
                        cat_ifvps = torch.cat(layer_ifvps, dim=0)

                        # Move to device
                        device_grads = self.strategy.move_to_device(cat_grads)
                        device_ifvps = self.strategy.move_to_device(cat_ifvps)

                        # Compute dot products for all samples at once
                        layer_influence = torch.sum(device_grads * device_ifvps, dim=1).cpu()

                        # Map results back to the original sample positions
                        start_idx = 0
                        for sample_start, sample_count in zip(layer_sample_starts, layer_sample_counts):
                            end_idx = start_idx + sample_count
                            self_influence[sample_start:sample_start + sample_count] += layer_influence[start_idx:end_idx]
                            start_idx = end_idx

                        # Clean up
                        del device_grads, device_ifvps, cat_grads, cat_ifvps, layer_influence
                        torch.cuda.empty_cache()

                    except RuntimeError as e:
                        logger.error(f"Error processing layer {layer_idx}: {e}")
                        # Fall back to per-batch processing if we run out of memory
                        for i, (grad, ifvp, sample_start, num_samples) in enumerate(
                                zip(layer_grads, layer_ifvps, layer_sample_starts, layer_sample_counts)):
                            try:
                                # Move to device
                                device_grad = self.strategy.move_to_device(grad)
                                device_ifvp = self.strategy.move_to_device(ifvp)

                                # Compute dot product
                                batch_influence = torch.sum(device_grad * device_ifvp, dim=1).cpu()

                                # Add to result
                                self_influence[sample_start:sample_start + num_samples] += batch_influence

                                # Clean up
                                del device_grad, device_ifvp, batch_influence
                                torch.cuda.empty_cache()
                            except RuntimeError as e:
                                logger.error(f"Error processing batch {i} in layer {layer_idx}: {e}")

                    # Clean up layer data
                    del layer_grads, layer_ifvps
                    torch.cuda.empty_cache()

                # Force cleanup after processing all layers for this chunk
                gc.collect()
                torch.cuda.empty_cache()
        else:
            # Direct processing (no dataloader) - use chunked processing
            logger.info("Processing batches directly...")

            # Process in chunks for memory efficiency
            chunk_size = dataloader_batch_size
            for chunk_start in tqdm(range(0, len(batch_indices), chunk_size), desc="Processing batch chunks"):
                chunk_end = min(chunk_start + chunk_size, len(batch_indices))
                chunk_batch_indices = batch_indices[chunk_start:chunk_end]

                # Process layers one at a time
                for layer_idx in range(len(self.layer_names)):
                    # Collect data for this layer
                    layer_grads = []
                    layer_ifvps = []
                    layer_sample_starts = []
                    layer_sample_counts = []

                    for batch_idx in chunk_batch_indices:
                        if batch_idx not in batch_to_sample_mapping:
                            continue

                        sample_start, sample_end = batch_to_sample_mapping[batch_idx]
                        num_samples = sample_end - sample_start

                        if num_samples <= 0:
                            continue

                        # Retrieve gradients and IFVP
                        grad_list = self.strategy.retrieve_gradients(batch_idx, is_test=False)
                        ifvp_list = self.strategy.retrieve_ifvp(batch_idx)

                        # Skip if layer is missing or empty
                        if (layer_idx >= len(grad_list) or grad_list[layer_idx].numel() == 0 or
                            layer_idx >= len(ifvp_list) or ifvp_list[layer_idx].numel() == 0):
                            continue

                        # Add to collection
                        layer_grads.append(grad_list[layer_idx])
                        layer_ifvps.append(ifvp_list[layer_idx])
                        layer_sample_starts.append(sample_start)
                        layer_sample_counts.append(num_samples)

                    # Skip if no data
                    if not layer_grads:
                        continue

                    # Process all batches for this layer together
                    try:
                        cat_grads = torch.cat(layer_grads, dim=0)
                        cat_ifvps = torch.cat(layer_ifvps, dim=0)

                        # Move to device
                        device_grads = self.strategy.move_to_device(cat_grads)
                        device_ifvps = self.strategy.move_to_device(cat_ifvps)

                        # Compute dot products efficiently
                        layer_influence = torch.sum(device_grads * device_ifvps, dim=1).cpu()

                        # Map results back
                        start_idx = 0
                        for sample_start, sample_count in zip(layer_sample_starts, layer_sample_counts):
                            end_idx = start_idx + sample_count
                            self_influence[sample_start:sample_start + sample_count] += layer_influence[start_idx:end_idx]
                            start_idx = end_idx

                        # Clean up
                        del device_grads, device_ifvps, cat_grads, cat_ifvps, layer_influence
                    except RuntimeError as e:
                        # Fallback to individual batch processing
                        logger.error(f"Error processing layer {layer_idx}: {e}")
                        for i, (grad, ifvp, sample_start, num_samples) in enumerate(
                                zip(layer_grads, layer_ifvps, layer_sample_starts, layer_sample_counts)):
                            try:
                                device_grad = self.strategy.move_to_device(grad)
                                device_ifvp = self.strategy.move_to_device(ifvp)
                                batch_influence = torch.sum(device_grad * device_ifvp, dim=1).cpu()
                                self_influence[sample_start:sample_start + num_samples] += batch_influence
                                del device_grad, device_ifvp, batch_influence
                            except RuntimeError as e:
                                logger.error(f"Error processing batch {i} in layer {layer_idx}: {e}")

                    # Clean up
                    del layer_grads, layer_ifvps
                    torch.cuda.empty_cache()

                # Force cleanup
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

        Optimized by:
        1. Processing dataloader chunks
        2. Processing each layer separately
        3. Concatenating all batches in a chunk for each layer

        Args:
            test_dataloader: DataLoader for test data
            train_dataloader: Optional DataLoader for training data if not cached
            use_cached_ifvp: Whether to use cached IFVP

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

        # Set up compressors if needed
        if self.sparsifiers is None and self.projectors is None:
            self._setup_compressors(test_dataloader)

        # Get or compute IFVP
        if use_cached_ifvp and self.strategy.has_ifvp():
            logger.info("Using cached IFVP")
        else:
            logger.info("Computing IFVP")
            self.compute_ifvp()

        # Compute test gradients
        logger.info("Computing test gradients")
        test_grads_dict, _ = self._compute_gradients(
            test_dataloader,
            is_test=True,
            batch_range=(0, len(test_dataloader))
        )

        # Ensure we clear any temp tensors
        torch.cuda.empty_cache()

        # Get batch mapping and determine total samples
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        total_train_samples = self.metadata.get_total_samples()

        # Process test batch indices and create test sample mapping
        logger.info("Organizing test data...")
        test_batch_indices = sorted(test_grads_dict.keys())
        test_sample_ranges = {}  # Maps batch_idx -> (start_idx, num_samples)
        current_index = 0

        # Create test sample mapping
        for test_batch_idx in test_batch_indices:
            batch_grads = test_grads_dict[test_batch_idx]
            batch_size = 0

            # Find first non-empty layer to determine batch size
            for layer_grads in batch_grads:
                if layer_grads.numel() > 0:
                    batch_size = layer_grads.shape[0]
                    break

            if batch_size == 0:
                logger.warning(f"Empty batch: test batch {test_batch_idx}")
                continue

            test_sample_ranges[test_batch_idx] = (current_index, batch_size)
            current_index += batch_size

        # Initialize influence scores tensor
        num_test = current_index  # Total number of test examples
        IF_score = torch.zeros(total_train_samples, num_test, device="cpu")

        # Remove hooks after collecting test gradients
        if self.hook_manager:
            self.hook_manager.remove_hooks()
            self.hook_manager = None

        # Create train IFVP dataloader with smaller batch size
        train_batch_size = 16
        ifvp_dataloader = self.strategy.create_gradient_dataloader(
            data_type="ifvp",
            batch_size=train_batch_size,
            num_workers=4,
            pin_memory=True
        )

        # Process one layer at a time
        logger.info("Processing influence scores layer by layer...")
        for layer_idx in tqdm(range(len(self.layer_names)), desc="Processing layers"):
            # First collect all test gradients for this layer
            test_layer_grads = []
            test_indices = []

            for test_batch_idx in test_batch_indices:
                if test_batch_idx not in test_sample_ranges:
                    continue

                # Skip if layer doesn't exist or is empty
                if (layer_idx >= len(test_grads_dict[test_batch_idx]) or
                    test_grads_dict[test_batch_idx][layer_idx].numel() == 0):
                    continue

                start_idx, num_samples = test_sample_ranges[test_batch_idx]
                grad = test_grads_dict[test_batch_idx][layer_idx]

                # Add to collection
                test_layer_grads.append(grad)
                test_indices.extend(range(start_idx, start_idx + num_samples))

            # Skip layer if no test gradients
            if not test_layer_grads:
                continue

            # Concatenate all test gradients for this layer
            cat_test_grads = torch.cat(test_layer_grads, dim=0)
            device_test_grads = self.strategy.move_to_device(cat_test_grads)

            # Process train batches in chunks
            if ifvp_dataloader:
                # Process using dataloader
                for train_batch_indices, train_ifvp_dicts in tqdm(
                    ifvp_dataloader,
                    desc=f"Processing layer {layer_idx}",
                    leave=False
                ):
                    # Skip empty batches
                    if not train_batch_indices:
                        continue

                    # Collect train IFVP data for this chunk
                    chunk_ifvps = []  # List of (ifvp, train_start, train_count)

                    for batch_pos, train_batch_idx in enumerate(train_batch_indices):
                        if train_batch_idx not in batch_to_sample_mapping:
                            continue

                        # Get sample range
                        train_start, train_end = batch_to_sample_mapping[train_batch_idx]
                        train_count = train_end - train_start

                        # Skip empty batches
                        if train_count <= 0:
                            continue

                        # Skip if layer doesn't exist or is empty
                        if (layer_idx not in train_ifvp_dicts[batch_pos] or
                            train_ifvp_dicts[batch_pos][layer_idx].numel() == 0):
                            continue

                        ifvp = train_ifvp_dicts[batch_pos][layer_idx]

                        # Add to collection
                        chunk_ifvps.append((ifvp, train_start, train_count))

                    # Skip if no valid train data
                    if not chunk_ifvps:
                        continue

                    # Process the entire chunk for this layer
                    try:
                        # Concatenate all IFVP data
                        cat_ifvps = []
                        train_ranges = []  # List of (train_start, train_count)

                        for ifvp, train_start, train_count in chunk_ifvps:
                            cat_ifvps.append(ifvp)
                            train_ranges.append((train_start, train_count))

                        # Concatenate and move to device
                        cat_ifvps = torch.cat(cat_ifvps, dim=0)
                        device_ifvps = self.strategy.move_to_device(cat_ifvps)

                        # Compute influence matrix: (train_samples x test_samples)
                        influence_matrix = torch.matmul(device_ifvps, device_test_grads.t()).cpu()

                        # Map results back to original positions
                        train_pos = 0
                        for train_start, train_count in train_ranges:
                            # For each test sample
                            for test_idx, orig_test_idx in enumerate(test_indices):
                                # Add influence scores
                                IF_score[train_start:train_start + train_count, orig_test_idx] += influence_matrix[
                                    train_pos:train_pos + train_count, test_idx
                                ]
                            train_pos += train_count

                        # Clean up
                        del device_ifvps, influence_matrix, cat_ifvps

                    except RuntimeError as e:
                        logger.error(f"Error processing chunk for layer {layer_idx}: {e}")
                        # Fall back to individual batch processing
                        for ifvp, train_start, train_count in chunk_ifvps:
                            try:
                                device_ifvp = self.strategy.move_to_device(ifvp)
                                influence = torch.matmul(device_ifvp, device_test_grads.t()).cpu()
                                for test_idx, orig_test_idx in enumerate(test_indices):
                                    IF_score[train_start:train_start + train_count, orig_test_idx] += influence[:, test_idx]
                                del device_ifvp, influence
                            except RuntimeError as e:
                                logger.error(f"Error processing individual batch: {e}")

                    # Force cleanup
                    torch.cuda.empty_cache()
            else:
                # Direct processing without dataloader - chunked approach
                logger.info(f"Processing layer {layer_idx} without dataloader")

                # Process train batches in chunks
                train_batch_indices = sorted(self.metadata.batch_info.keys())
                chunk_size = train_batch_size
                for chunk_start in tqdm(
                    range(0, len(train_batch_indices), chunk_size),
                    desc=f"Processing layer {layer_idx} chunks",
                    leave=False
                ):
                    chunk_end = min(chunk_start + chunk_size, len(train_batch_indices))
                    chunk_batch_indices = train_batch_indices[chunk_start:chunk_end]

                    # Collect IFVP data for this chunk
                    chunk_ifvps = []  # List of (ifvp, train_start, train_count)

                    for train_batch_idx in chunk_batch_indices:
                        if train_batch_idx not in batch_to_sample_mapping:
                            continue

                        # Get sample range
                        train_start, train_end = batch_to_sample_mapping[train_batch_idx]
                        train_count = train_end - train_start

                        # Skip empty batches
                        if train_count <= 0:
                            continue

                        # Retrieve IFVP
                        train_ifvp_list = self.strategy.retrieve_ifvp(train_batch_idx)

                        # Skip if layer doesn't exist or is empty
                        if (layer_idx >= len(train_ifvp_list) or
                            train_ifvp_list[layer_idx].numel() == 0):
                            continue

                        ifvp = train_ifvp_list[layer_idx]

                        # Add to collection
                        chunk_ifvps.append((ifvp, train_start, train_count))

                    # Skip if no valid data
                    if not chunk_ifvps:
                        continue

                    # Process the chunk
                    try:
                        # Concatenate all IFVP data
                        cat_ifvps = []
                        train_ranges = []

                        for ifvp, train_start, train_count in chunk_ifvps:
                            cat_ifvps.append(ifvp)
                            train_ranges.append((train_start, train_count))

                        cat_ifvps = torch.cat(cat_ifvps, dim=0)
                        device_ifvps = self.strategy.move_to_device(cat_ifvps)

                        # Compute influence matrix efficiently
                        influence_matrix = torch.matmul(device_ifvps, device_test_grads.t()).cpu()

                        # Map results back
                        train_pos = 0
                        for train_start, train_count in train_ranges:
                            for test_idx, orig_test_idx in enumerate(test_indices):
                                IF_score[train_start:train_start + train_count, orig_test_idx] += influence_matrix[
                                    train_pos:train_pos + train_count, test_idx
                                ]
                            train_pos += train_count

                        # Clean up
                        del device_ifvps, influence_matrix, cat_ifvps

                    except RuntimeError as e:
                        # Fall back to per-batch processing
                        logger.error(f"Error processing chunk for layer {layer_idx}: {e}")
                        for ifvp, train_start, train_count in chunk_ifvps:
                            try:
                                device_ifvp = self.strategy.move_to_device(ifvp)
                                influence = torch.matmul(device_ifvp, device_test_grads.t()).cpu()
                                for test_idx, orig_test_idx in enumerate(test_indices):
                                    IF_score[train_start:train_start + train_count, orig_test_idx] += influence[:, test_idx]
                                del device_ifvp, influence
                            except RuntimeError as e:
                                logger.error(f"Error processing individual batch: {e}")

                    # Force cleanup
                    torch.cuda.empty_cache()

            # Clean up test gradients for this layer
            del device_test_grads, cat_test_grads
            gc.collect()
            torch.cuda.empty_cache()

        # Return result
        if self.profile and self.profiling_stats:
            return (IF_score, self.profiling_stats)
        else:
            return IF_score