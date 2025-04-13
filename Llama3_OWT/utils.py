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

def count_total_tokens(dataloader):
    """Count the total number of tokens in a dataloader."""
    total_tokens = 0
    for batch in dataloader:
        # Count non-padding tokens in input_ids
        # Assuming attention_mask indicates which tokens are padding (1 for real tokens, 0 for padding)
        if "attention_mask" in batch:
            total_tokens += batch["attention_mask"].sum().item()
        else:
            # If no attention mask, count all tokens
            total_tokens += batch["input_ids"].numel()
    return total_tokens

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
            if proj_method == "SJLT":
                assert int(proj_dim[0]) > 512, "Projection dimension must be greater than 512 for to project the entire gradient to avoid the slow down due to local collisions."
        else:
            proj_factorize = False
            proj_dim = int(proj_dim) # Convert to integer

    # Compatibility checking
    if proj_method == "Localize":
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

def batch_size(baseline, tda):
    if baseline == "GC":
        if tda == "GD":
            train_batch_size = 2
            test_batch_size = 2
        elif tda in ["IF-NONE", "IF-RAW", "IF-KFAC", "IF-EKFAC"]:
            train_batch_size = 12
            test_batch_size = 12
    elif baseline == "LoGra":
        if tda in ["IF-NONE", "IF-RAW", "IF-KFAC", "IF-EKFAC"]:
            train_batch_size = 2
            test_batch_size = 2
    elif baseline == "LogIX":
        if tda in ["IF-NONE", "IF-RAW", "IF-KFAC", "IF-EKFAC"]:
            train_batch_size = 32
            test_batch_size = 32
    else:
        raise ValueError("Invalid baseline and tda combination.")

    return train_batch_size, test_batch_size

def result_filename(args):
    filename_parts = []

    if args.projection is not None:
        filename_parts.append(args.projection)


    filename_parts.append(f"thrd-{args.threshold}")
    filename_parts.append(f"rdp-{args.random_drop}")

    # Join parts and save the file
    result_filename = f"./results/{args.baseline}/{args.tda}/{args.layer}/{'_'.join(filename_parts)}.pt"

    return result_filename