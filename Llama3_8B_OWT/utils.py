from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator, List

import torch
from torch.utils.data import Sampler

import numpy as np
from scipy.stats import spearmanr
import csv

class SubsetSampler(Sampler):
    """Samples elements from a predefined list of indices.

    Note that for training, the built-in PyTorch
    SubsetRandomSampler should be used. This class is for
    attributting process.
    """

    def __init__(self, indices: List[int]) -> None:
        """Initialize the sampler.

        Args:
            indices (list): A list of indices to sample from.
        """
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        """Get an iterator for the sampler.

        Returns:
            An iterator for the sampler.
        """
        return iter(self.indices)

    def __len__(self) -> int:
        """Get the number of indices in the sampler.

        Returns:
            The number of indices in the sampler.
        """
        return len(self.indices)

def get_worker_batch_range(train_dataloader, worker_arg="1/1"):
    """
    Parse the worker argument in format "{worker_id}/{total_workers}" and
    get the batch range for the specified worker.

    Args:
        train_dataloader: DataLoader for the training data
        worker_arg: String in format "{worker_id}/{total_workers}" (default: "1/1")

    Returns:
        Tuple of (worker_id, (start_batch, end_batch)) for the specified worker
    """
    try:
        # Parse the worker argument
        parts = worker_arg.split('/')
        if len(parts) != 2:
            raise ValueError("Worker argument must be in format 'worker_id/total_workers'")

        worker_id = int(parts[0]) - 1  # Convert to 0-indexed
        num_workers = int(parts[1])

        # Validate parsed values
        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(f"worker_id must be between 1 and {num_workers}")
        if num_workers <= 0:
            raise ValueError("total_workers must be greater than 0")

        # Calculate total number of batches
        total_batches = len(train_dataloader)

        # Calculate batch range size for each worker
        batch_size_per_worker = total_batches // num_workers
        remaining_batches = total_batches % num_workers

        # Calculate start_batch for the specific worker
        start_batch = worker_id * batch_size_per_worker
        # Add adjustment for workers that get an extra batch (if total doesn't divide evenly)
        start_batch += min(worker_id, remaining_batches)

        # Calculate worker_batch_size
        worker_batch_size = batch_size_per_worker + (1 if worker_id < remaining_batches else 0)

        # Calculate end_batch
        end_batch = start_batch + worker_batch_size

        return (worker_id, (start_batch, end_batch))

    except Exception as e:
        print(f"Error parsing worker argument: {e}")
        return None

def setup_projection_kwargs(args, device):
    if args.projection is None:
        proj_method = "Identity"
        proj_factorize = False
        proj_dim = -1
    else:
        proj_method, proj_dim = args.projection.split("-")

        # proj_dim might be of the form 'proj_dim*proj_dim' for factorized projection, for simply 'proj_dim' for non-factorized projection
        if "*" in proj_dim:
            proj_factorize = True
            proj_dim = proj_dim.split("*")
            assert proj_dim[0] == proj_dim[1], "Projection dimension must be the same for factorized projection."

            proj_dim = int(proj_dim[0]) # Convert to integer
        else:
            proj_factorize = False
            proj_dim = int(proj_dim) # Convert to integer

    # Compatibility checking
    if proj_method == "Localize":
        assert args.baseline == "GC", "Localize option only works with GC baseline."
        assert args.layer == "Linear", "Localize option only works with Linear layer."
        assert args.random_drop == 0.0, "Localize option can't be combined with random drop."

    projector_kwargs = {
        "proj_dim": proj_dim,
        "proj_max_batch_size": 32,
        "proj_seed": args.seed,
        "proj_factorize": proj_factorize,
        "device": device,
        "method": proj_method,
        "use_half_precision": False,
        "threshold": args.threshold,
        "random_drop": args.random_drop,
    }

    return projector_kwargs

def result_filename(args):
    filename_parts = []

    if args.projection is not None:
        filename_parts.append(args.projection)


    filename_parts.append(f"thrd-{args.threshold}")
    filename_parts.append(f"rdp-{args.random_drop}")

    # Join parts and save the file
    result_filename = f"./results/{args.baseline}/{args.tda}/{args.layer}/{'_'.join(filename_parts)}.pt"

    return result_filename