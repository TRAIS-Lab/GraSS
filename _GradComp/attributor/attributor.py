"""
Enhanced concrete implementation of the Influence Function Attributor with chunked I/O optimization.
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
    Enhanced Influence function calculator with chunked I/O and optimized batch processing.
    Uses deterministic chunking with flexible worker-based batch range allocation.
    """

    def compute_preconditioners(self, damping: Optional[float] = None) -> List[torch.Tensor]:
        """Compute preconditioners (inverse Hessian) from gradients using chunked processing."""
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

        # Use chunked dataloader for efficient processing
        dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=4,
            pin_memory=True
        )

        if dataloader:
            logger.info("Processing gradients using chunked dataloader")

            # Process chunks deterministically
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
                        aggregated_gradients = torch.cat(layer_gradients, dim=0)
                        aggregated_gradients = self.strategy.move_to_device(aggregated_gradients)

                        sample_counts[layer_idx] += aggregated_gradients.shape[0]

                        # Compute Hessian contribution efficiently
                        batch_hessian = torch.matmul(aggregated_gradients.t(), aggregated_gradients)

                        # Update the Hessian accumulator
                        if hessian_accumulators[layer_idx] is None:
                            hessian_accumulators[layer_idx] = batch_hessian
                        else:
                            hessian_accumulators[layer_idx] += batch_hessian

                        del aggregated_gradients, batch_hessian

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning(f"GPU memory overflow for layer {layer_idx}, processing individually")
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

                    del layer_gradients
                    torch.cuda.empty_cache()

                gc.collect()
                torch.cuda.empty_cache()

        else:
            # Fallback to direct processing
            logger.info("Processing gradients using direct batch access")
            batch_indices = sorted(self.metadata.batch_info.keys())

            for chunk_start in tqdm(range(0, len(batch_indices), self.chunk_size),
                                  desc="Computing preconditioners"):
                chunk_end = min(chunk_start + self.chunk_size, len(batch_indices))
                chunk_batch_indices = batch_indices[chunk_start:chunk_end]

                for layer_idx in range(len(self.layer_names)):
                    layer_gradients = []

                    for batch_idx in chunk_batch_indices:
                        batch_grads = self.strategy.retrieve_gradients(batch_idx, is_test=False)
                        if layer_idx < len(batch_grads) and batch_grads[layer_idx].numel() > 0:
                            layer_gradients.append(batch_grads[layer_idx])

                    if not layer_gradients:
                        continue

                    # Process layer gradients in fixed sub-batches
                    grad_batch_size = 50
                    for i in range(0, len(layer_gradients), grad_batch_size):
                        batch_grads = layer_gradients[i:i+grad_batch_size]

                        try:
                            combined_gradients = torch.cat(batch_grads, dim=0)
                            combined_gradients = self.strategy.move_to_device(combined_gradients)

                            batch_hessian = torch.matmul(combined_gradients.t(), combined_gradients)

                            if hessian_accumulators[layer_idx] is None:
                                hessian_accumulators[layer_idx] = batch_hessian
                            else:
                                hessian_accumulators[layer_idx] += batch_hessian

                            sample_counts[layer_idx] += combined_gradients.shape[0]

                            del combined_gradients, batch_hessian

                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                for grad in batch_grads:
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
                torch.cuda.empty_cache()

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
        Compute inverse-Hessian-vector products (IFVP) using chunked processing.
        Automatically determines batch range from worker specification and metadata.
        """
        logger.info(f"Worker {worker}: Computing IFVP with chunked processing")

        # Clear IFVP cache first to avoid conflicts with previous computations
        # self.clear_ifvp_cache()

        # Load batch information if needed
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found. Call cache_gradients first or ensure metadata exists.")

        # Calculate total batches from metadata instead of dataloader
        batch_indices = sorted(self.metadata.batch_info.keys())
        if not batch_indices:
            raise ValueError("No batches found in metadata")

        # Calculate total batches based on the range of batch indices in metadata
        min_batch_idx = min(batch_indices)
        max_batch_idx = max(batch_indices)
        total_batches = max_batch_idx + 1  # Assuming batch indices start from 0

        logger.info(f"Found {len(batch_indices)} batches in metadata (range: {min_batch_idx}-{max_batch_idx})")

        # Calculate worker batch range using metadata-derived total
        start_batch, end_batch = self._get_worker_batch_range(total_batches, worker)

        # Start batch range processing for disk offload
        if self.offload == "disk" and hasattr(self.strategy, 'start_batch_range_processing'):
            self.strategy.start_batch_range_processing(start_batch, end_batch)

        logger.info(f"Processing batch range: [{start_batch}, {end_batch}) ({end_batch - start_batch} batches)")

        # Return raw gradients if Hessian type is "none"
        if self.hessian == "none":
            logger.info("Using raw gradients as IFVP since hessian type is 'none'")
            return self._copy_gradients_to_ifvp(start_batch, end_batch)

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Get batch indices for this worker from metadata
        worker_batch_indices = [idx for idx in batch_indices
                            if start_batch <= idx < end_batch]

        if not worker_batch_indices:
            logger.warning(f"No batches found in range [{start_batch}, {end_batch}) for worker {worker}")
            return {}

        logger.info(f"Processing {len(worker_batch_indices)} batches for this worker: {worker_batch_indices[:5]}{'...' if len(worker_batch_indices) > 5 else ''}")

        # Pre-load all preconditioners
        logger.info("Pre-loading preconditioners for all layers")
        preconditioners = []
        for layer_idx in range(len(self.layer_names)):
            precond = self.strategy.retrieve_preconditioner(layer_idx)
            preconditioners.append(precond)

        valid_preconditioners = sum(1 for p in preconditioners if p is not None)
        logger.info(f"Loaded {valid_preconditioners} preconditioners out of {len(self.layer_names)} layers")

        # Use chunked dataloader for efficient processing
        dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=2,
            pin_memory=True,
            batch_range=(start_batch, end_batch)
        )

        result_dict = {}

        if dataloader:
            logger.info("Processing IFVP using chunked dataloader")

            # Process chunks deterministically
            for chunk_data in tqdm(dataloader, desc="Computing IFVP from chunks"):
                batch_indices_chunk, batch_grad_dicts = chunk_data
                batch_ifvp_dicts = [{} for _ in range(len(batch_indices_chunk))]

                # Process each layer across all batches in the chunk
                for layer_idx in range(len(self.layer_names)):
                    precond = preconditioners[layer_idx]
                    if precond is None:
                        for i in range(len(batch_indices_chunk)):
                            batch_ifvp_dicts[i][layer_idx] = torch.tensor([])
                        continue

                    # Collect gradients and track their batch positions
                    layer_gradients = []
                    gradient_batch_map = []

                    for batch_dict_idx, batch_grad_dict in enumerate(batch_grad_dicts):
                        if layer_idx not in batch_grad_dict or batch_grad_dict[layer_idx].numel() == 0:
                            batch_ifvp_dicts[batch_dict_idx][layer_idx] = torch.tensor([])
                            continue

                        layer_gradients.append(batch_grad_dict[layer_idx])
                        gradient_batch_map.append(batch_dict_idx)

                    if not layer_gradients:
                        continue

                    try:
                        # Move to device and concatenate for efficient processing
                        device_gradients = [self.strategy.move_to_device(grad) for grad in layer_gradients]
                        original_shapes = [grad.shape[0] for grad in device_gradients]
                        concatenated_gradients = torch.cat(device_gradients, dim=0)

                        # Move preconditioner to device and ensure type compatibility
                        device_precond = self.strategy.move_to_device(precond)
                        device_precond = device_precond.to(dtype=concatenated_gradients.dtype)

                        # Compute IFVP efficiently for all gradients at once
                        ifvp_result = torch.matmul(device_precond, concatenated_gradients.t()).t()

                        # Split results back to individual batches
                        split_results = torch.split(ifvp_result, original_shapes)

                        # Distribute results back to batch dictionaries
                        for i, (split_result, dict_idx) in enumerate(zip(split_results, gradient_batch_map)):
                            batch_ifvp_dicts[dict_idx][layer_idx] = self.strategy.move_from_device(split_result)

                        del device_gradients, concatenated_gradients, ifvp_result, split_results, device_precond

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning(f"GPU memory overflow for layer {layer_idx}, processing individually")
                            for grad, dict_idx in zip(layer_gradients, gradient_batch_map):
                                device_grad = self.strategy.move_to_device(grad)
                                device_precond = self.strategy.move_to_device(precond)
                                device_precond = device_precond.to(dtype=device_grad.dtype)
                                ifvp = torch.matmul(device_precond, device_grad.t()).t()
                                batch_ifvp_dicts[dict_idx][layer_idx] = self.strategy.move_from_device(ifvp)
                                del device_grad, device_precond, ifvp
                        else:
                            raise

                    torch.cuda.empty_cache()

                # Store computed IFVP results
                for i, (batch_idx, batch_ifvp_dict) in enumerate(zip(batch_indices_chunk, batch_ifvp_dicts)):
                    # Convert to list for consistent interface
                    batch_ifvp_list = []
                    for layer_idx in range(len(self.layer_names)):
                        if layer_idx in batch_ifvp_dict:
                            batch_ifvp_list.append(batch_ifvp_dict[layer_idx])
                        else:
                            batch_ifvp_list.append(torch.tensor([]))

                    # Store in strategy
                    self.strategy.store_ifvp(batch_idx, batch_ifvp_list)
                    result_dict[batch_idx] = batch_ifvp_list

        else:
            # Fallback to direct processing using metadata batch indices
            logger.info("Processing IFVP using direct batch access")

            for chunk_start in tqdm(range(0, len(worker_batch_indices), self.chunk_size),
                                desc="Computing IFVP"):
                chunk_end = min(chunk_start + self.chunk_size, len(worker_batch_indices))
                chunk_batch_indices = worker_batch_indices[chunk_start:chunk_end]

                for batch_idx in chunk_batch_indices:
                    batch_ifvp = [torch.tensor([]) for _ in range(len(self.layer_names))]
                    batch_grads = self.strategy.retrieve_gradients(batch_idx, is_test=False)

                    # Process each layer
                    for layer_idx in range(len(self.layer_names)):
                        if (layer_idx >= len(batch_grads) or
                            batch_grads[layer_idx].numel() == 0 or
                            preconditioners[layer_idx] is None):
                            continue

                        # Compute IFVP
                        grad = self.strategy.move_to_device(batch_grads[layer_idx])
                        device_precond = self.strategy.move_to_device(preconditioners[layer_idx])
                        device_precond = device_precond.to(dtype=grad.dtype)

                        ifvp = torch.matmul(device_precond, grad.t()).t()
                        batch_ifvp[layer_idx] = self.strategy.move_from_device(ifvp)

                        del grad, ifvp, device_precond
                        torch.cuda.empty_cache()

                    # Store results
                    self.strategy.store_ifvp(batch_idx, batch_ifvp)
                    result_dict[batch_idx] = batch_ifvp

        # Finalize batch range processing for disk offload
        if self.offload == "disk" and hasattr(self.strategy, 'finish_batch_range_processing'):
            self.strategy.finish_batch_range_processing()

        # Clean up preloaded preconditioners
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
        # Use metadata to get actual batch indices instead of assuming continuous range
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
        """Compute self-influence scores using optimized chunked processing."""
        logger.info(f"Worker {worker}: Computing self-influence scores")

        # Load batch information if needed
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found. Call cache_gradients first.")

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

        batch_indices = [idx for idx in self.metadata.batch_info.keys()
                        if start_batch <= idx < end_batch]

        # Process batches in fixed chunks
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

    """
Enhanced concrete implementation of the Influence Function Attributor with chunked I/O optimization.
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
    Enhanced Influence function calculator with chunked I/O and optimized batch processing.
    Uses deterministic chunking with flexible worker-based batch range allocation.
    """

    def compute_preconditioners(self, damping: Optional[float] = None) -> List[torch.Tensor]:
        """Compute preconditioners (inverse Hessian) from gradients using chunked processing."""
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

        # Use chunked dataloader for efficient processing
        dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=4,
            pin_memory=True
        )

        if dataloader:
            logger.info("Processing gradients using chunked dataloader")

            # Process chunks deterministically
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
                        aggregated_gradients = torch.cat(layer_gradients, dim=0)
                        aggregated_gradients = self.strategy.move_to_device(aggregated_gradients)

                        sample_counts[layer_idx] += aggregated_gradients.shape[0]

                        # Compute Hessian contribution efficiently
                        batch_hessian = torch.matmul(aggregated_gradients.t(), aggregated_gradients)

                        # Update the Hessian accumulator
                        if hessian_accumulators[layer_idx] is None:
                            hessian_accumulators[layer_idx] = batch_hessian
                        else:
                            hessian_accumulators[layer_idx] += batch_hessian

                        del aggregated_gradients, batch_hessian

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning(f"GPU memory overflow for layer {layer_idx}, processing individually")
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

                    del layer_gradients
                    torch.cuda.empty_cache()

                gc.collect()
                torch.cuda.empty_cache()

        else:
            # Fallback to direct processing
            logger.info("Processing gradients using direct batch access")
            batch_indices = sorted(self.metadata.batch_info.keys())

            for chunk_start in tqdm(range(0, len(batch_indices), self.chunk_size),
                                  desc="Computing preconditioners"):
                chunk_end = min(chunk_start + self.chunk_size, len(batch_indices))
                chunk_batch_indices = batch_indices[chunk_start:chunk_end]

                for layer_idx in range(len(self.layer_names)):
                    layer_gradients = []

                    for batch_idx in chunk_batch_indices:
                        batch_grads = self.strategy.retrieve_gradients(batch_idx, is_test=False)
                        if layer_idx < len(batch_grads) and batch_grads[layer_idx].numel() > 0:
                            layer_gradients.append(batch_grads[layer_idx])

                    if not layer_gradients:
                        continue

                    # Process layer gradients in fixed sub-batches
                    grad_batch_size = 50
                    for i in range(0, len(layer_gradients), grad_batch_size):
                        batch_grads = layer_gradients[i:i+grad_batch_size]

                        try:
                            combined_gradients = torch.cat(batch_grads, dim=0)
                            combined_gradients = self.strategy.move_to_device(combined_gradients)

                            batch_hessian = torch.matmul(combined_gradients.t(), combined_gradients)

                            if hessian_accumulators[layer_idx] is None:
                                hessian_accumulators[layer_idx] = batch_hessian
                            else:
                                hessian_accumulators[layer_idx] += batch_hessian

                            sample_counts[layer_idx] += combined_gradients.shape[0]

                            del combined_gradients, batch_hessian

                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                for grad in batch_grads:
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
                torch.cuda.empty_cache()

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
        Compute inverse-Hessian-vector products (IFVP) using chunked processing.
        Automatically determines batch range from worker specification and metadata.
        """
        logger.info(f"Worker {worker}: Computing IFVP with chunked processing")

        # Clear IFVP cache first to avoid conflicts with previous computations
        # self.clear_ifvp_cache()

        # Load batch information if needed
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found. Call cache_gradients first or ensure metadata exists.")

        # Calculate total batches from metadata instead of dataloader
        batch_indices = sorted(self.metadata.batch_info.keys())
        if not batch_indices:
            raise ValueError("No batches found in metadata")

        # Calculate total batches based on the range of batch indices in metadata
        min_batch_idx = min(batch_indices)
        max_batch_idx = max(batch_indices)
        total_batches = max_batch_idx + 1  # Assuming batch indices start from 0

        logger.info(f"Found {len(batch_indices)} batches in metadata (range: {min_batch_idx}-{max_batch_idx})")

        # Calculate worker batch range using metadata-derived total
        start_batch, end_batch = self._get_worker_batch_range(total_batches, worker)

        # Start batch range processing for disk offload
        if self.offload == "disk" and hasattr(self.strategy, 'start_batch_range_processing'):
            self.strategy.start_batch_range_processing(start_batch, end_batch)

        logger.info(f"Processing batch range: [{start_batch}, {end_batch}) ({end_batch - start_batch} batches)")

        # Return raw gradients if Hessian type is "none"
        if self.hessian == "none":
            logger.info("Using raw gradients as IFVP since hessian type is 'none'")
            return self._copy_gradients_to_ifvp(start_batch, end_batch)

        if self.profile and self.profiling_stats:
            torch.cuda.synchronize(self.device)
            start_time = time.time()

        # Get batch indices for this worker from metadata
        worker_batch_indices = [idx for idx in batch_indices
                            if start_batch <= idx < end_batch]

        if not worker_batch_indices:
            logger.warning(f"No batches found in range [{start_batch}, {end_batch}) for worker {worker}")
            return {}

        logger.info(f"Processing {len(worker_batch_indices)} batches for this worker: {worker_batch_indices[:5]}{'...' if len(worker_batch_indices) > 5 else ''}")

        # Pre-load all preconditioners
        logger.info("Pre-loading preconditioners for all layers")
        preconditioners = []
        for layer_idx in range(len(self.layer_names)):
            precond = self.strategy.retrieve_preconditioner(layer_idx)
            preconditioners.append(precond)

        valid_preconditioners = sum(1 for p in preconditioners if p is not None)
        logger.info(f"Loaded {valid_preconditioners} preconditioners out of {len(self.layer_names)} layers")

        # Use chunked dataloader for efficient processing
        dataloader = self.strategy.create_gradient_dataloader(
            data_type="gradients",
            batch_size=2,
            pin_memory=True,
            batch_range=(start_batch, end_batch)
        )

        result_dict = {}

        if dataloader:
            logger.info("Processing IFVP using chunked dataloader")

            # Process chunks deterministically
            for chunk_data in tqdm(dataloader, desc="Computing IFVP from chunks"):
                batch_indices_chunk, batch_grad_dicts = chunk_data
                batch_ifvp_dicts = [{} for _ in range(len(batch_indices_chunk))]

                # Process each layer across all batches in the chunk
                for layer_idx in range(len(self.layer_names)):
                    precond = preconditioners[layer_idx]
                    if precond is None:
                        for i in range(len(batch_indices_chunk)):
                            batch_ifvp_dicts[i][layer_idx] = torch.tensor([])
                        continue

                    # Collect gradients and track their batch positions
                    layer_gradients = []
                    gradient_batch_map = []

                    for batch_dict_idx, batch_grad_dict in enumerate(batch_grad_dicts):
                        if layer_idx not in batch_grad_dict or batch_grad_dict[layer_idx].numel() == 0:
                            batch_ifvp_dicts[batch_dict_idx][layer_idx] = torch.tensor([])
                            continue

                        layer_gradients.append(batch_grad_dict[layer_idx])
                        gradient_batch_map.append(batch_dict_idx)

                    if not layer_gradients:
                        continue

                    try:
                        # Move to device and concatenate for efficient processing
                        device_gradients = [self.strategy.move_to_device(grad) for grad in layer_gradients]
                        original_shapes = [grad.shape[0] for grad in device_gradients]
                        concatenated_gradients = torch.cat(device_gradients, dim=0)

                        # Move preconditioner to device and ensure type compatibility
                        device_precond = self.strategy.move_to_device(precond)
                        device_precond = device_precond.to(dtype=concatenated_gradients.dtype)

                        # Compute IFVP efficiently for all gradients at once
                        ifvp_result = torch.matmul(device_precond, concatenated_gradients.t()).t()

                        # Split results back to individual batches
                        split_results = torch.split(ifvp_result, original_shapes)

                        # Distribute results back to batch dictionaries
                        for i, (split_result, dict_idx) in enumerate(zip(split_results, gradient_batch_map)):
                            batch_ifvp_dicts[dict_idx][layer_idx] = self.strategy.move_from_device(split_result)

                        del device_gradients, concatenated_gradients, ifvp_result, split_results, device_precond

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning(f"GPU memory overflow for layer {layer_idx}, processing individually")
                            for grad, dict_idx in zip(layer_gradients, gradient_batch_map):
                                device_grad = self.strategy.move_to_device(grad)
                                device_precond = self.strategy.move_to_device(precond)
                                device_precond = device_precond.to(dtype=device_grad.dtype)
                                ifvp = torch.matmul(device_precond, device_grad.t()).t()
                                batch_ifvp_dicts[dict_idx][layer_idx] = self.strategy.move_from_device(ifvp)
                                del device_grad, device_precond, ifvp
                        else:
                            raise

                    torch.cuda.empty_cache()

                # Store computed IFVP results
                for i, (batch_idx, batch_ifvp_dict) in enumerate(zip(batch_indices_chunk, batch_ifvp_dicts)):
                    # Convert to list for consistent interface
                    batch_ifvp_list = []
                    for layer_idx in range(len(self.layer_names)):
                        if layer_idx in batch_ifvp_dict:
                            batch_ifvp_list.append(batch_ifvp_dict[layer_idx])
                        else:
                            batch_ifvp_list.append(torch.tensor([]))

                    # Store in strategy
                    self.strategy.store_ifvp(batch_idx, batch_ifvp_list)
                    result_dict[batch_idx] = batch_ifvp_list

        else:
            # Fallback to direct processing using metadata batch indices
            logger.info("Processing IFVP using direct batch access")

            for chunk_start in tqdm(range(0, len(worker_batch_indices), self.chunk_size),
                                desc="Computing IFVP"):
                chunk_end = min(chunk_start + self.chunk_size, len(worker_batch_indices))
                chunk_batch_indices = worker_batch_indices[chunk_start:chunk_end]

                for batch_idx in chunk_batch_indices:
                    batch_ifvp = [torch.tensor([]) for _ in range(len(self.layer_names))]
                    batch_grads = self.strategy.retrieve_gradients(batch_idx, is_test=False)

                    # Process each layer
                    for layer_idx in range(len(self.layer_names)):
                        if (layer_idx >= len(batch_grads) or
                            batch_grads[layer_idx].numel() == 0 or
                            preconditioners[layer_idx] is None):
                            continue

                        # Compute IFVP
                        grad = self.strategy.move_to_device(batch_grads[layer_idx])
                        device_precond = self.strategy.move_to_device(preconditioners[layer_idx])
                        device_precond = device_precond.to(dtype=grad.dtype)

                        ifvp = torch.matmul(device_precond, grad.t()).t()
                        batch_ifvp[layer_idx] = self.strategy.move_from_device(ifvp)

                        del grad, ifvp, device_precond
                        torch.cuda.empty_cache()

                    # Store results
                    self.strategy.store_ifvp(batch_idx, batch_ifvp)
                    result_dict[batch_idx] = batch_ifvp

        # Finalize batch range processing for disk offload
        if self.offload == "disk" and hasattr(self.strategy, 'finish_batch_range_processing'):
            self.strategy.finish_batch_range_processing()

        # Clean up preloaded preconditioners
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
        # Use metadata to get actual batch indices instead of assuming continuous range
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
        """Compute self-influence scores using optimized chunked processing."""
        logger.info(f"Worker {worker}: Computing self-influence scores")

        # Load batch information if needed
        if not self.metadata.batch_info:
            logger.info("Loading batch information from metadata...")
            self.metadata._load_metadata_if_exists()

        if not self.metadata.batch_info:
            raise ValueError("No batch information found. Call cache_gradients first.")

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

        batch_indices = [idx for idx in self.metadata.batch_info.keys()
                        if start_batch <= idx < end_batch]

        # Process batches in fixed chunks
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
        """Attribute influence using optimized chunked processing with efficient data loading."""
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

        # Set up compressors if needed
        if self.sparsifiers is None and self.projectors is None:
            self._setup_compressors(test_dataloader)

        # Get or compute IFVP
        if use_cached_ifvp and self.strategy.has_ifvp():
            logger.info("Using cached IFVP")
        else:
            logger.info("Computing IFVP")
            self.compute_ifvp()

        # Compute and store ALL test gradients in memory (test data is usually small)
        logger.info("Computing and caching all test gradients in memory")
        test_grads_dict, _ = self._compute_gradients_chunked(
            test_dataloader,
            is_test=True,
            worker="0/1"  # Always process all test data
        )

        # Organize test data efficiently
        logger.info("Organizing test data for efficient processing")
        test_batch_indices = sorted(test_grads_dict.keys())
        all_test_gradients = []  # List of (layer_idx, tensor) for all test samples
        test_sample_mapping = {}  # Maps test_batch_idx to (start_idx, end_idx) in flattened array
        current_test_idx = 0

        # Flatten all test gradients by layer for efficient processing
        num_test_samples = 0
        for test_batch_idx in test_batch_indices:
            batch_grads = test_grads_dict[test_batch_idx]
            batch_size = 0

            # Find batch size from first non-empty layer
            for layer_grads in batch_grads:
                if layer_grads.numel() > 0:
                    batch_size = layer_grads.shape[0]
                    break

            if batch_size > 0:
                test_sample_mapping[test_batch_idx] = (current_test_idx, current_test_idx + batch_size)
                current_test_idx += batch_size
                num_test_samples += batch_size

        logger.info(f"Total test samples: {num_test_samples}")

        # Pre-organize test gradients by layer for efficient matrix operations
        test_gradients_by_layer = []
        for layer_idx in range(len(self.layer_names)):
            layer_test_grads = []

            for test_batch_idx in test_batch_indices:
                batch_grads = test_grads_dict[test_batch_idx]
                if layer_idx < len(batch_grads) and batch_grads[layer_idx].numel() > 0:
                    layer_test_grads.append(batch_grads[layer_idx])

            if layer_test_grads:
                # Concatenate all test gradients for this layer
                concatenated_test_grad = torch.cat(layer_test_grads, dim=0)
                test_gradients_by_layer.append(concatenated_test_grad)
            else:
                test_gradients_by_layer.append(None)

        # Get batch mappings
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        total_train_samples = self.metadata.get_total_samples()

        # Initialize influence score matrix
        IF_score = torch.zeros(total_train_samples, num_test_samples, device="cpu")

        # Create efficient dataloader for training IFVP data
        logger.info("Creating efficient dataloader for training data")
        train_ifvp_dataloader = self.strategy.create_gradient_dataloader(
            data_type="ifvp",
            batch_size=4,  # Larger batch size for efficiency
            pin_memory=True
        )

        if train_ifvp_dataloader:
            logger.info("Processing attribution using chunked IFVP dataloader")

            # Process training data in chunks using dataloader
            for chunk_data in tqdm(train_ifvp_dataloader, desc="Computing attribution from chunks"):
                train_batch_indices_chunk, train_ifvp_dicts = chunk_data

                # Process each layer with concatenated chunk data
                for layer_idx in range(len(self.layer_names)):
                    if test_gradients_by_layer[layer_idx] is None:
                        continue

                    # Collect all IFVP tensors and their sample ranges for this layer
                    layer_ifvp_tensors = []
                    sample_ranges = []

                    for batch_idx, train_batch_ifvp_dict in zip(train_batch_indices_chunk, train_ifvp_dicts):
                        if (batch_idx in batch_to_sample_mapping and
                            layer_idx in train_batch_ifvp_dict and
                            train_batch_ifvp_dict[layer_idx].numel() > 0):

                            train_start, train_end = batch_to_sample_mapping[batch_idx]
                            if train_end > train_start:
                                layer_ifvp_tensors.append(train_batch_ifvp_dict[layer_idx])
                                sample_ranges.append((train_start, train_end))

                    if not layer_ifvp_tensors:
                        continue

                    # Concatenate all IFVP tensors for this layer in the chunk
                    concatenated_ifvp = torch.cat(layer_ifvp_tensors, dim=0)
                    concatenated_ifvp = self.strategy.move_to_device(concatenated_ifvp)
                    test_grad_tensor = self.strategy.move_to_device(test_gradients_by_layer[layer_idx])

                    # Single large matrix multiplication for the entire chunk and layer
                    chunk_influence_matrix = torch.matmul(concatenated_ifvp, test_grad_tensor.t()).cpu()

                    # Distribute results back to the correct positions in IF_score
                    current_row = 0
                    for train_start, train_end in sample_ranges:
                        batch_size = train_end - train_start
                        IF_score[train_start:train_end, :] += chunk_influence_matrix[current_row:current_row + batch_size, :]
                        current_row += batch_size

                    del concatenated_ifvp, test_grad_tensor, chunk_influence_matrix
                    torch.cuda.empty_cache()

        else:
            # Fallback to direct processing (less efficient but still better than original)
            logger.info("Processing attribution using direct batch access")
            train_batch_indices = sorted(self.metadata.batch_info.keys())

            for train_chunk_start in tqdm(range(0, len(train_batch_indices), self.chunk_size),
                                        desc="Computing attribution"):
                train_chunk_end = min(train_chunk_start + self.chunk_size, len(train_batch_indices))
                train_chunk_batch_indices = train_batch_indices[train_chunk_start:train_chunk_end]

                for train_batch_idx in train_chunk_batch_indices:
                    if train_batch_idx not in batch_to_sample_mapping:
                        continue

                    train_start, train_end = batch_to_sample_mapping[train_batch_idx]
                    train_count = train_end - train_start

                    if train_count <= 0:
                        continue

                    # Load training IFVP once for this batch
                    train_ifvp = self.strategy.retrieve_ifvp(train_batch_idx)

                    # Process each layer efficiently
                    for layer_idx in range(len(self.layer_names)):
                        if (layer_idx >= len(train_ifvp) or train_ifvp[layer_idx].numel() == 0 or
                            test_gradients_by_layer[layer_idx] is None):
                            continue

                        # Move data to device
                        train_ifvp_tensor = self.strategy.move_to_device(train_ifvp[layer_idx])
                        test_grad_tensor = self.strategy.move_to_device(test_gradients_by_layer[layer_idx])

                        # Efficient matrix multiplication for all test samples at once
                        influence_matrix = torch.matmul(train_ifvp_tensor, test_grad_tensor.t()).cpu()

                        # Add to influence score matrix
                        IF_score[train_start:train_end, :] += influence_matrix

                        del train_ifvp_tensor, test_grad_tensor, influence_matrix
                        torch.cuda.empty_cache()

        # Clean up test gradients from memory
        del test_grads_dict, test_gradients_by_layer
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"Attribution computation completed. Result shape: {IF_score.shape}")

        if self.profile and self.profiling_stats:
            return (IF_score, self.profiling_stats)
        else:
            return IF_score