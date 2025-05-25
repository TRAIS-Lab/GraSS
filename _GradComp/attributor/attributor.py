"""
Influence Function Attributor.
"""

from typing import TYPE_CHECKING, Optional, Union, Tuple
import time

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

import torch
from tqdm import tqdm

from .base import BaseAttributor, ProfilingStats, ProcessingInfo
from ..utils.common import stable_inverse

import logging
logger = logging.getLogger(__name__)

class IFAttributor(BaseAttributor):
    """
    Influence function calculator with optimized I/O managing.
    """

    def compute_preconditioners(self, damping: Optional[float] = None) -> Union[ProcessingInfo, Tuple[ProcessingInfo, ProfilingStats]]:
        """Compute preconditioners (inverse Hessian) from gradients using tensor-based processing."""
        logger.info(f"Computing preconditioners with hessian type: {self.hessian}")

        if damping is None:
            damping = self.damping

        # Load batch information if needed
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found. Call cache_gradients first.")

        # Synchronize layer dimensions
        self._sync_layer_dims()

        if self.layer_dims is None:
            raise ValueError("Layer dimensions not found. Ensure gradients have been computed and stored.")

        logger.info(f"Computing preconditioners for {len(self.layer_names)} layers")

        # If hessian type is "none", no preconditioners needed
        if self.hessian == "none":
            logger.info("Hessian type is 'none', skipping preconditioner computation")
            for layer_idx in range(len(self.layer_names)):
                self.strategy.store_preconditioner(layer_idx, None)

            # Create processing info
            processing_info = ProcessingInfo(
                num_batches=0,
                total_samples=0,
                batch_range=(0, 0),
                data_type="preconditioners"
            )

            if self.profile and self.profiling_stats:
                return (processing_info, self.profiling_stats)
            else:
                return processing_info

        total_samples = self.metadata.get_total_samples()
        logger.info(f"Computing preconditioners from {total_samples} total samples")

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Initialize Hessian accumulators for all layers
        hessian_accumulators = [None] * len(self.layer_names)
        sample_counts = [0] * len(self.layer_names)

        # Use tensor-based dataloader for efficient processing
        logger.debug("Using tensor-based dataloader for preconditioner computation")

        dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=4,  # Process 4 chunks at a time
            pin_memory=True
        )

        if dataloader:
            for chunk_tensor, batch_mapping in tqdm(dataloader, desc="Computing preconditioners from chunks"):
                # Move chunk to device
                chunk_tensor = self.strategy.move_to_device(chunk_tensor)

                # Process each layer
                for layer_idx in range(len(self.layer_names)):
                    # Extract layer slice
                    start_col = sum(self.layer_dims[:layer_idx])
                    end_col = start_col + self.layer_dims[layer_idx]
                    layer_data = chunk_tensor[:, start_col:end_col]

                    if layer_data.numel() == 0:
                        continue

                    sample_counts[layer_idx] += layer_data.shape[0]

                    # Compute Hessian contribution for this chunk
                    batch_hessian = torch.matmul(layer_data.t(), layer_data)

                    # Accumulate to the Hessian
                    if hessian_accumulators[layer_idx] is None:
                        hessian_accumulators[layer_idx] = batch_hessian
                    else:
                        hessian_accumulators[layer_idx] += batch_hessian

                    del batch_hessian, layer_data

                del chunk_tensor
                torch.cuda.empty_cache()


        # Compute preconditioners from accumulated Hessians
        computed_count = 0
        for layer_idx in tqdm(range(len(self.layer_names)), desc="Computing inverses"):
            hessian_accumulator = hessian_accumulators[layer_idx]
            sample_count = sample_counts[layer_idx]

            if hessian_accumulator is not None and sample_count > 0:
                # Normalize by total number of samples
                hessian = hessian_accumulator / sample_count

                # Compute inverse based on Hessian type
                if self.hessian == "raw":
                    precond = stable_inverse(hessian, damping=damping)
                    self.strategy.store_preconditioner(layer_idx, precond)
                    del precond

                elif self.hessian in ["kfac", "ekfac"]:
                    self.strategy.store_preconditioner(layer_idx, hessian) #TODO: Fix, currently not correct

                computed_count += 1
                del hessian_accumulator, hessian
                torch.cuda.empty_cache()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

        self.strategy.wait_for_async_operations()

        # Create processing info
        processing_info = ProcessingInfo(
            num_batches=computed_count,
            total_samples=total_samples,
            batch_range=(0, len(self.layer_names)),
            data_type="preconditioners"
        )

        if self.profile and self.profiling_stats:
            return (processing_info, self.profiling_stats)
        else:
            return processing_info

    def compute_ifvp(self, worker: str = "0/1") -> Union[ProcessingInfo, Tuple[ProcessingInfo, ProfilingStats]]:
        """
        Compute inverse-Hessian-vector products (IFVP) and store in strategy.

        Returns:
            ProcessingInfo about what was computed
        """
        logger.info(f"Worker {worker}: Computing IFVP with tensor-based processing")

        # Load batch information if needed
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found.")

        # Synchronize layer dimensions
        self._sync_layer_dims()

        if self.layer_dims is None:
            raise ValueError("Layer dimensions not found. Ensure gradients have been computed and stored.")

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
            logger.debug("Using raw gradients as IFVP since hessian type is 'none'")
            return self._copy_gradients_as_ifvp(start_batch, end_batch)

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Load all preconditioners once
        preconditioners = []
        for layer_idx in range(len(self.layer_names)):
            precond = self.strategy.retrieve_preconditioner(layer_idx)
            preconditioners.append(precond)

        valid_preconditioners = sum(1 for p in preconditioners if p is not None)
        logger.debug(f"Loaded {valid_preconditioners} preconditioners out of {len(self.layer_names)} layers")

        # Get batch mapping
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        processed_batches = 0
        processed_samples = 0

        # Use tensor-based dataloader
        logger.debug("Processing IFVP using tensor-based dataloader")

        dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=4,  # Process 4 chunks at a time
            pin_memory=True,
            batch_range=(start_batch, end_batch)
        )

        if dataloader:
            for chunk_tensor, batch_mapping in tqdm(dataloader, desc="Computing IFVP from chunks"):
                # Move chunk to device
                chunk_tensor = self.strategy.move_to_device(chunk_tensor)

                # Process each batch in the chunk
                for batch_idx, (start_row, end_row) in batch_mapping.items():
                    if batch_idx not in batch_to_sample_mapping:
                        continue

                    batch_tensor = chunk_tensor[start_row:end_row]
                    batch_ifvp = []

                    # Process each layer
                    for layer_idx in range(len(self.layer_names)):
                        if preconditioners[layer_idx] is None:
                            batch_ifvp.append(torch.zeros(batch_tensor.shape[0], self.layer_dims[layer_idx]))
                            continue

                        # Extract layer data
                        start_col = sum(self.layer_dims[:layer_idx])
                        end_col = start_col + self.layer_dims[layer_idx]
                        layer_grad = batch_tensor[:, start_col:end_col]

                        if layer_grad.numel() == 0:
                            batch_ifvp.append(torch.zeros(batch_tensor.shape[0], self.layer_dims[layer_idx]))
                            continue

                        # Get preconditioner
                        device_precond = self.strategy.move_to_device(preconditioners[layer_idx])
                        device_precond = device_precond.to(dtype=layer_grad.dtype)

                        # Compute IFVP: H^{-1} @ g
                        ifvp = torch.matmul(device_precond, layer_grad.t()).t()
                        batch_ifvp.append(ifvp)

                        del layer_grad, ifvp, device_precond

                    # Store IFVP for this batch using strategy
                    self.strategy.store_ifvp(batch_idx, batch_ifvp)

                    processed_batches += 1
                    processed_samples += batch_tensor.shape[0]

                    del batch_ifvp

                del chunk_tensor
                torch.cuda.empty_cache()

        # Finalize batch range processing
        if self.offload == "disk" and hasattr(self.strategy, 'finish_batch_range_processing'):
            self.strategy.finish_batch_range_processing()

        # Clean up
        del preconditioners
        torch.cuda.empty_cache()

        self.strategy.wait_for_async_operations()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

        # Create processing info
        processing_info = ProcessingInfo(
            num_batches=processed_batches,
            total_samples=processed_samples,
            batch_range=(start_batch, end_batch),
            data_type="ifvp"
        )

        if self.profile and self.profiling_stats:
            return (processing_info, self.profiling_stats)
        else:
            return processing_info

    def _copy_gradients_as_ifvp(self, start_batch: int, end_batch: int) -> Union[ProcessingInfo, Tuple[ProcessingInfo, ProfilingStats]]:
        """Copy gradients as IFVP when hessian type is 'none'."""
        # Ensure layer dimensions are loaded
        if self.layer_dims is None:
            self._sync_layer_dims()

        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        processed_batches = 0
        processed_samples = 0

        # Process using tensor dataloader
        dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=1,
            pin_memory=True,
            batch_range=(start_batch, end_batch)
        )

        if dataloader:
            for chunk_tensor, batch_mapping in tqdm(dataloader, desc="Copying gradients as IFVP"):
                for batch_idx, (start_row, end_row) in batch_mapping.items():
                    if batch_idx not in batch_to_sample_mapping:
                        continue

                    # Extract batch and split into layers
                    batch_tensor = chunk_tensor[start_row:end_row]
                    gradients = []

                    for layer_idx in range(len(self.layer_names)):
                        start_col = sum(self.layer_dims[:layer_idx])
                        end_col = start_col + self.layer_dims[layer_idx]
                        gradients.append(batch_tensor[:, start_col:end_col].contiguous())

                    self.strategy.store_ifvp(batch_idx, gradients)

                    processed_batches += 1
                    processed_samples += batch_tensor.shape[0]

                    del gradients

                del chunk_tensor
                torch.cuda.empty_cache()

        # Create processing info
        processing_info = ProcessingInfo(
            num_batches=processed_batches,
            total_samples=processed_samples,
            batch_range=(start_batch, end_batch),
            data_type="ifvp"
        )

        if self.profile and self.profiling_stats:
            return (processing_info, self.profiling_stats)
        else:
            return processing_info

    @torch.no_grad()
    def compute_self_influence(self, worker: str = "0/1") -> Union[torch.Tensor, Tuple[torch.Tensor, ProfilingStats]]:
        """Compute self-influence scores using tensor-based processing."""
        logger.info(f"Worker {worker}: Computing self-influence scores")

        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found.")

        # Synchronize layer dimensions
        self._sync_layer_dims()

        if self.layer_dims is None:
            raise ValueError("Layer dimensions not found. Ensure gradients have been computed and stored.")

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

        # Use tensor dataloaders for both gradients and IFVP
        grad_dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=4,
            pin_memory=True,
            batch_range=(start_batch, end_batch)
        )

        ifvp_dataloader = self.strategy.create_gradient_dataloader(
            data_type="ifvp",
            batch_size=4,
            pin_memory=True,
            batch_range=(start_batch, end_batch)
        )

        if grad_dataloader and ifvp_dataloader:
            # Process in parallel
            for (grad_tensor, grad_mapping), (ifvp_tensor, ifvp_mapping) in tqdm(
                zip(grad_dataloader, ifvp_dataloader),
                desc="Computing self-influence",
                total=len(grad_dataloader)
            ):
                # Move to device
                grad_tensor = self.strategy.move_to_device(grad_tensor)
                ifvp_tensor = self.strategy.move_to_device(ifvp_tensor)

                # Process each batch
                for batch_idx in grad_mapping:
                    if batch_idx not in batch_to_sample_mapping or batch_idx not in ifvp_mapping:
                        continue

                    sample_start, sample_end = batch_to_sample_mapping[batch_idx]

                    # Extract batch slices
                    grad_start, grad_end = grad_mapping[batch_idx]
                    ifvp_start, ifvp_end = ifvp_mapping[batch_idx]

                    batch_grad = grad_tensor[grad_start:grad_end]
                    batch_ifvp = ifvp_tensor[ifvp_start:ifvp_end]

                    # Compute dot product
                    batch_influence = torch.sum(batch_grad * batch_ifvp, dim=1).cpu()
                    self_influence[sample_start:sample_end] = batch_influence

                del grad_tensor, ifvp_tensor
                torch.cuda.empty_cache()

        if self.profile and self.profiling_stats:
            return (self_influence, self.profiling_stats)
        else:
            return self_influence

    def attribute(
        self,
        test_dataloader: 'DataLoader',
        train_dataloader: Optional['DataLoader'] = None,
        use_cached_ifvp: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ProfilingStats]]:
        """
        Attribute influence using efficient single-pass tensor-based processing.
        """
        logger.info("Computing influence attribution")

        # Load batch information if needed
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            if train_dataloader is None:
                raise ValueError("No batch information found and no training dataloader provided.")
            logger.info("No batch metadata found. Caching gradients from provided dataloader...")
            self.cache_gradients(train_dataloader)

        # Synchronize layer dimensions
        self._sync_layer_dims()

        if self.layer_dims is None:
            raise ValueError("Layer dimensions not found. Ensure gradients have been computed and stored.")

        # Set up compressors if needed
        if self.sparsifiers is None and self.projectors is None:
            self._setup_compressors(test_dataloader)

        # Get or compute IFVP
        if use_cached_ifvp and self.strategy.has_ifvp():
            logger.info("Using cached IFVP")
        else:
            logger.info("Computing IFVP")
            self.compute_ifvp()

        # Compute test gradients once
        logger.info("Computing test gradients")
        test_grads_tensor, test_batch_mapping = self._compute_gradients_for_batches(
            test_dataloader,
            start_batch=0,
            end_batch=len(test_dataloader),
            is_test=True
        )

        if test_grads_tensor is None or test_grads_tensor.numel() == 0:
            logger.warning("No test gradients computed")
            test_sample_count = 0
            all_test_gradients = torch.empty(0, 0)
        else:
            test_sample_count = test_grads_tensor.shape[0]
            all_test_gradients = test_grads_tensor
            logger.debug(f"Test data shape: {all_test_gradients.shape}")

        # Get batch mappings
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        total_train_samples = self.metadata.get_total_samples()

        # Initialize result
        IF_score = torch.zeros(total_train_samples, test_sample_count, device=self.device)

        if test_sample_count == 0:
            logger.warning("No test samples, returning zero influence scores")
            if self.profile and self.profiling_stats:
                return (IF_score, self.profiling_stats)
            else:
                return IF_score

        # Start profiling
        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Create dataloader for IFVP with optimal batch size
        train_ifvp_dataloader = self.strategy.create_gradient_dataloader(
            data_type="ifvp",
            batch_size=2,
            pin_memory=True
        )

        if train_ifvp_dataloader:
            logger.info("Starting efficient double-batched attribution computation")

            # Configure test batching for memory efficiency
            test_batch_size = min(32, test_sample_count)  # Process test samples in chunks
            logger.debug(f"Using test batch size: {test_batch_size}")

            # Single pass through training IFVP data with nested test batching
            for chunk_tensor, batch_mapping in tqdm(train_ifvp_dataloader, desc="Computing attribution"):
                # Move train chunk to device
                chunk_tensor_device = self.strategy.move_to_device(chunk_tensor)

                # Process test gradients in batches to save memory
                for test_start in range(0, test_sample_count, test_batch_size):
                    test_end = min(test_start + test_batch_size, test_sample_count)
                    test_batch = all_test_gradients[test_start:test_end]

                    # Move test batch to device
                    test_batch_device = self.strategy.move_to_device(test_batch)

                    # Efficient batched matrix multiplication for this (train_chunk, test_batch) pair
                    # Shape: (chunk_samples, proj_dim) @ (proj_dim, test_batch_samples) -> (chunk_samples, test_batch_samples)
                    chunk_scores = torch.matmul(chunk_tensor_device, test_batch_device.t())

                    # Map chunk results back to global sample indices
                    for batch_idx, (start_row, end_row) in batch_mapping.items():
                        if batch_idx not in batch_to_sample_mapping:
                            continue

                        train_start, train_end = batch_to_sample_mapping[batch_idx]
                        batch_scores = chunk_scores[start_row:end_row]
                        IF_score[train_start:train_end, test_start:test_end] = batch_scores.to(IF_score.device)

                    # Clean up test batch from device
                    del test_batch_device, chunk_scores
                    torch.cuda.empty_cache()

                # Clean up train chunk from device
                del chunk_tensor_device
                torch.cuda.empty_cache()

        else:
            logger.warning("No IFVP dataloader available, attribution may be incomplete")

        # Clean up
        del all_test_gradients
        torch.cuda.empty_cache()

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            self.profiling_stats.precondition += time.time() - start_time

        logger.info(f"Attribution computation completed. Result shape: {IF_score.shape}")

        if self.profile and self.profiling_stats:
            return (IF_score, self.profiling_stats)
        else:
            return IF_score