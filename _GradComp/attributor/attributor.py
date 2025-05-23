"""
Enhanced concrete implementation of the Influence Function Attributor with tensor-based optimization.
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
    Enhanced Influence function calculator with tensor-based I/O and optimized batch processing.
    Uses efficient tensor operations for maximum performance.
    """

    def compute_preconditioners(self, damping: Optional[float] = None) -> List[torch.Tensor]:
        """Compute preconditioners (inverse Hessian) from gradients using chunk-by-chunk processing."""
        logger.info(f"Computing preconditioners with hessian type: {self.hessian}")

        if damping is None:
            damping = self.damping

        # Load batch information if needed
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found. Call cache_gradients first.")

        # If hessian type is "none", no preconditioners needed
        if self.hessian == "none":
            logger.info("Hessian type is 'none', skipping preconditioner computation")
            for layer_idx in range(len(self.layer_names)):
                self.strategy.store_preconditioner(layer_idx, None)
            return [None] * len(self.layer_names)

        total_samples = self.metadata.get_total_samples()
        logger.info(f"Computing preconditioners from {total_samples} total samples")

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Initialize Hessian accumulators for all layers
        hessian_accumulators = [None] * len(self.layer_names)
        sample_counts = [0] * len(self.layer_names)

        # Use chunk-based dataloader for efficient processing
        logger.info("Using chunk-based dataloader for preconditioner computation")

        dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=4,  # Process 4 chunks at a time
            pin_memory=True
        )

        if dataloader:
            for chunk_data in tqdm(dataloader, desc="Computing preconditioners from chunks"):
                batch_indices, batch_dicts = chunk_data

                # Process each layer across all batches in the chunk
                for layer_idx in range(len(self.layer_names)):
                    layer_gradients = []

                    # Collect gradients for this layer from all batches in chunk
                    for batch_dict in batch_dicts:
                        if layer_idx not in batch_dict or batch_dict[layer_idx].numel() == 0:
                            continue
                        layer_gradients.append(batch_dict[layer_idx])

                    if not layer_gradients:
                        continue

                    try:
                        # Concatenate gradients from this chunk
                        aggregated_gradients = torch.cat(layer_gradients, dim=0)
                        aggregated_gradients = self.strategy.move_to_device(aggregated_gradients)

                        sample_counts[layer_idx] += aggregated_gradients.shape[0]

                        # Compute Hessian contribution for this chunk
                        batch_hessian = torch.matmul(aggregated_gradients.t(), aggregated_gradients)

                        # Accumulate to the Hessian
                        if hessian_accumulators[layer_idx] is None:
                            hessian_accumulators[layer_idx] = batch_hessian
                        else:
                            hessian_accumulators[layer_idx] += batch_hessian

                        del aggregated_gradients, batch_hessian

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning(f"GPU memory overflow for layer {layer_idx}, processing smaller batches")
                            # Process gradients one by one
                            for grad in layer_gradients:
                                grad_device = self.strategy.move_to_device(grad)
                                sample_counts[layer_idx] += grad_device.shape[0]
                                small_hessian = torch.matmul(grad_device.t(), grad_device)
                                if hessian_accumulators[layer_idx] is None:
                                    hessian_accumulators[layer_idx] = small_hessian
                                else:
                                    hessian_accumulators[layer_idx] += small_hessian
                                del grad_device, small_hessian
                        else:
                            raise

                    torch.cuda.empty_cache()

                gc.collect()

        # Compute preconditioners from accumulated Hessians
        preconditioners = [None] * len(self.layer_names)

        for layer_idx in tqdm(range(len(self.layer_names)), desc="Computing inverses"):
            hessian_accumulator = hessian_accumulators[layer_idx]
            sample_count = sample_counts[layer_idx]

            if hessian_accumulator is not None and sample_count > 0:
                # Normalize by total number of samples
                hessian = hessian_accumulator / sample_count

                # Compute inverse based on Hessian type
                if self.hessian == "raw":
                    precond = stable_inverse(hessian, damping=damping)
                    preconditioners[layer_idx] = precond
                    self.strategy.store_preconditioner(layer_idx, precond)
                    del precond

                elif self.hessian in ["kfac", "ekfac"]:
                    preconditioners[layer_idx] = hessian
                    self.strategy.store_preconditioner(layer_idx, hessian)

                del hessian_accumulator, hessian
                torch.cuda.empty_cache()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

        self.strategy.wait_for_async_operations()

        if self.profile and self.profiling_stats:
            return (preconditioners, self.profiling_stats)
        else:
            return preconditioners

    def compute_ifvp(self, worker: str = "0/1") -> Dict[int, List[torch.Tensor]]:
        """
        Compute inverse-Hessian-vector products (IFVP) using chunk-based processing.
        """
        logger.info(f"Worker {worker}: Computing IFVP with chunk-based processing")

        # Load batch information if needed
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found.")

        # Calculate batch range
        batch_indices = sorted(self.metadata.batch_info.keys())
        min_batch_idx = min(batch_indices)
        max_batch_idx = max(batch_indices)
        total_batches = max_batch_idx + 1

        start_batch, end_batch = self._get_worker_batch_range(total_batches, worker)

        # Start batch range processing
        if self.offload == "disk" and hasattr(self.strategy, 'start_batch_range_processing'):
            self.strategy.start_batch_range_processing(start_batch, end_batch)

        logger.info(f"Processing batch range: [{start_batch}, {end_batch})")

        # Return raw gradients if Hessian type is "none"
        if self.hessian == "none":
            logger.info("Using raw gradients as IFVP since hessian type is 'none'")
            return self._copy_gradients_to_ifvp(start_batch, end_batch)

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Load all preconditioners once
        preconditioners = []
        for layer_idx in range(len(self.layer_names)):
            precond = self.strategy.retrieve_preconditioner(layer_idx)
            preconditioners.append(precond)

        valid_preconditioners = sum(1 for p in preconditioners if p is not None)
        logger.info(f"Loaded {valid_preconditioners} preconditioners out of {len(self.layer_names)} layers")

        result_dict = {}

        # Use chunk-based dataloader
        logger.info("Processing IFVP using chunk-based dataloader")

        dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=2,  # Process 2 chunks at a time
            pin_memory=True,
            batch_range=(start_batch, end_batch)
        )

        if dataloader:
            for chunk_data in tqdm(dataloader, desc="Computing IFVP from chunks"):
                batch_indices_chunk, batch_grad_dicts = chunk_data

                # Process each batch in the chunk
                for batch_idx, batch_grad_dict in zip(batch_indices_chunk, batch_grad_dicts):
                    batch_ifvp = []

                    # Process each layer
                    for layer_idx in range(len(self.layer_names)):
                        if (layer_idx not in batch_grad_dict or
                            batch_grad_dict[layer_idx].numel() == 0 or
                            preconditioners[layer_idx] is None):
                            batch_ifvp.append(torch.tensor([]))
                            continue

                        # Get gradient and preconditioner
                        grad = self.strategy.move_to_device(batch_grad_dict[layer_idx])
                        device_precond = self.strategy.move_to_device(preconditioners[layer_idx])
                        device_precond = device_precond.to(dtype=grad.dtype)

                        # Compute IFVP: H^{-1} @ g
                        ifvp = torch.matmul(device_precond, grad.t()).t()
                        batch_ifvp.append(self.strategy.move_from_device(ifvp))

                        del grad, ifvp, device_precond
                        torch.cuda.empty_cache()

                    # Store IFVP for this batch
                    self.strategy.store_ifvp(batch_idx, batch_ifvp)
                    result_dict[batch_idx] = batch_ifvp

        # Finalize batch range processing
        if self.offload == "disk" and hasattr(self.strategy, 'finish_batch_range_processing'):
            self.strategy.finish_batch_range_processing()

        # Clean up
        del preconditioners
        gc.collect()
        torch.cuda.empty_cache()

        self.strategy.wait_for_async_operations()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time
            return (result_dict, self.profiling_stats)
        else:
            return result_dict

    def _copy_gradients_to_ifvp(self, start_batch: int, end_batch: int) -> Dict[int, List[torch.Tensor]]:
        """Copy gradients to IFVP storage when hessian type is 'none'."""
        batch_indices = [idx for idx in self.metadata.batch_info.keys()
                        if start_batch <= idx < end_batch]

        result_dict = {}
        for chunk_start in tqdm(range(0, len(batch_indices), self.chunk_size),
                            desc="Copying gradients to IFVP"):
            chunk_end = min(chunk_start + self.chunk_size, len(batch_indices))
            chunk_batch_indices = batch_indices[chunk_start:chunk_end]

            for batch_idx in chunk_batch_indices:
                gradients = self.strategy.retrieve_gradients(batch_idx, is_test=False)
                self.strategy.store_ifvp(batch_idx, gradients)
                result_dict[batch_idx] = None

        return result_dict

    def compute_self_influence(self, worker: str = "0/1") -> torch.Tensor:
        """Compute self-influence scores using tensor-based processing."""
        logger.info(f"Worker {worker}: Computing self-influence scores")

        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found.")

        # Make sure IFVP is computed
        if not self.strategy.has_ifvp():
            logger.info("IFVP not found, computing it now...")
            self.compute_ifvp(worker=worker)

        # Get batch mapping and worker range
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        total_samples = self.metadata.get_total_samples()
        self_influence = torch.zeros(total_samples, device="cpu")

        # Get worker batch range
        total_batches = len(self.full_train_dataloader)
        start_batch, end_batch = self._get_worker_batch_range(total_batches, worker)

        # Try tensor-based computation
        if hasattr(self.strategy, 'load_all_as_tensor'):
            logger.info("Using tensor-based self-influence computation")

            # Load gradients and IFVP as tensors
            grad_tensor, grad_mapping = self.strategy.load_all_as_tensor("gradients", (start_batch, end_batch))
            ifvp_tensor, ifvp_mapping = self.strategy.load_all_as_tensor("ifvp", (start_batch, end_batch))

            if grad_tensor.numel() > 0 and ifvp_tensor.numel() > 0:
                # Ensure mappings are aligned
                assert grad_mapping.keys() == ifvp_mapping.keys(), "Gradient and IFVP mappings don't match"

                # Compute element-wise product and sum
                influence_per_sample = torch.sum(grad_tensor * ifvp_tensor, dim=1).cpu()

                # Map back to sample indices
                for batch_idx in grad_mapping:
                    if batch_idx in batch_to_sample_mapping:
                        sample_start, sample_end = batch_to_sample_mapping[batch_idx]
                        grad_start, grad_end = grad_mapping[batch_idx]

                        self_influence[sample_start:sample_end] = influence_per_sample[grad_start:grad_end]
        else:
            # Fallback to batch-based processing
            batch_indices = [idx for idx in self.metadata.batch_info.keys()
                            if start_batch <= idx < end_batch]

            for chunk_start in tqdm(range(0, len(batch_indices), self.chunk_size),
                                  desc="Computing self-influence"):
                chunk_end = min(chunk_start + self.chunk_size, len(batch_indices))
                chunk_batch_indices = batch_indices[chunk_start:chunk_end]

                for batch_idx in chunk_batch_indices:
                    if batch_idx not in batch_to_sample_mapping:
                        continue

                    sample_start, sample_end = batch_to_sample_mapping[batch_idx]
                    num_samples = sample_end - sample_start

                    if num_samples <= 0:
                        continue

                    # Load gradients and IFVP
                    gradients = self.strategy.retrieve_gradients(batch_idx, is_test=False)
                    ifvp = self.strategy.retrieve_ifvp(batch_idx)

                    # Compute self-influence for each layer
                    for layer_idx in range(len(self.layer_names)):
                        if (layer_idx >= len(gradients) or gradients[layer_idx].numel() == 0 or
                            layer_idx >= len(ifvp) or ifvp[layer_idx].numel() == 0):
                            continue

                        grad = self.strategy.move_to_device(gradients[layer_idx])
                        ifvp_tensor = self.strategy.move_to_device(ifvp[layer_idx])

                        # Compute dot product
                        layer_influence = torch.sum(grad * ifvp_tensor, dim=1).cpu()
                        self_influence[sample_start:sample_start + num_samples] += layer_influence

                        del grad, ifvp_tensor, layer_influence
                        torch.cuda.empty_cache()

        return self_influence

    def attribute(
        self,
        test_dataloader: 'DataLoader',
        train_dataloader: Optional['DataLoader'] = None,
        use_cached_ifvp: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ProfilingStats]]:
        """Attribute influence using tensor-based processing for maximum efficiency."""
        logger.info("Computing influence attribution with tensor-based optimization")

        # Load batch information if needed
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            if train_dataloader is None:
                raise ValueError("No batch information found and no training dataloader provided.")
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

        # Compute test gradients - FIXED METHOD NAME AND UNPACKING
        logger.info("Computing test gradients")
        test_grads_tensor, test_batch_mapping = self._compute_gradients_direct(
            test_dataloader,
            is_test=True,
            worker="0/1"
        )

        # test_grads_tensor contains the concatenated test gradients
        # test_batch_mapping contains the mapping from batch_idx to (start_row, end_row)

        if test_grads_tensor is None or test_grads_tensor.numel() == 0:
            logger.warning("No test gradients computed")
            test_sample_count = 0
            all_test_gradients = torch.empty(0, 0)
        else:
            test_sample_count = test_grads_tensor.shape[0]
            all_test_gradients = test_grads_tensor
            logger.info(f"Test data shape: {all_test_gradients.shape}")

        # Get batch mappings
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        total_train_samples = self.metadata.get_total_samples()

        # Initialize result
        IF_score = torch.zeros(total_train_samples, test_sample_count, device=self.device)

        # Try tensor-based computation for maximum efficiency
        if hasattr(self.strategy, 'load_all_as_tensor'):
            logger.info("Using tensor-based attribution computation")

            # Load all IFVP data as tensor
            ifvp_tensor, batch_mapping = self.strategy.load_all_as_tensor("ifvp")

            if ifvp_tensor.numel() > 0 and all_test_gradients.numel() > 0:
                # Move to device
                ifvp_device = self.strategy.move_to_device(ifvp_tensor)
                test_device = self.strategy.move_to_device(all_test_gradients)

                # Single matrix multiplication for all attributions
                attribution_matrix = torch.matmul(ifvp_device, test_device.t())

                # Map back to sample indices
                for batch_idx, (start_row, end_row) in batch_mapping.items():
                    if batch_idx in batch_to_sample_mapping:
                        sample_start, sample_end = batch_to_sample_mapping[batch_idx]
                        IF_score[sample_start:sample_end, :] = attribution_matrix[start_row:end_row, :]

                del ifvp_device, test_device, attribution_matrix
                torch.cuda.empty_cache()
        else:
            # Fallback to chunk-based processing
            logger.info("Using chunk-based attribution computation")

            # Get layer dimensions
            layer_dims = self.layer_dims

            # Create dataloader for IFVP
            train_ifvp_dataloader = self.strategy.create_gradient_dataloader(
                data_type="ifvp",
                batch_size=2,
                pin_memory=True
            )

            if train_ifvp_dataloader:
                for chunk_data in tqdm(train_ifvp_dataloader, desc="Computing attribution"):
                    train_batch_indices, train_ifvp_dicts = chunk_data

                    # Process chunk
                    chunk_tensors = []
                    chunk_ranges = []

                    for batch_idx, ifvp_dict in zip(train_batch_indices, train_ifvp_dicts):
                        if batch_idx not in batch_to_sample_mapping:
                            continue

                        train_start, train_end = batch_to_sample_mapping[batch_idx]

                        # Concatenate layers
                        batch_concat = []
                        for layer_idx in range(len(self.layer_names)):
                            if layer_idx in ifvp_dict and ifvp_dict[layer_idx].numel() > 0:
                                batch_concat.append(ifvp_dict[layer_idx])
                            else:
                                # Zero tensor
                                if layer_dims and layer_idx < len(layer_dims):
                                    dim = layer_dims[layer_idx]
                                else:
                                    dim = 0
                                batch_size = train_end - train_start
                                batch_concat.append(torch.zeros(batch_size, dim))

                        if batch_concat:
                            chunk_tensors.append(torch.cat(batch_concat, dim=1))
                            chunk_ranges.append((train_start, train_end))

                    if chunk_tensors:
                        # Process chunk
                        chunk_ifvp = torch.cat(chunk_tensors, dim=0)
                        chunk_ifvp_device = self.strategy.move_to_device(chunk_ifvp)
                        test_device = self.strategy.move_to_device(all_test_gradients)

                        chunk_scores = torch.matmul(chunk_ifvp_device, test_device.t())

                        # Map back
                        row_offset = 0
                        for train_start, train_end in chunk_ranges:
                            batch_size = train_end - train_start
                            IF_score[train_start:train_end, :] = chunk_scores[row_offset:row_offset + batch_size, :]
                            row_offset += batch_size

                        del chunk_ifvp_device, test_device, chunk_scores

        # Clean up
        del all_test_gradients
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"Attribution computation completed. Result shape: {IF_score.shape}")

        if self.profile and self.profiling_stats:
            return (IF_score, self.profiling_stats)
        else:
            return IF_score