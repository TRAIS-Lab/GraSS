from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator, List

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from torch.utils.data import Sampler

import numpy as np
from scipy.stats import spearmanr
import csv

def replace_conv1d_modules(model):
    # GPT-2 is defined in terms of Conv1D. However, this does not work for EK-FAC.
    # Here, we convert these Conv1D modules to linear modules recursively.
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_conv1d_modules(module)

        if isinstance(module, Conv1D):
            new_module = nn.Linear(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
            )
            new_module.weight.data.copy_(module.weight.data.t())
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)
    return model

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
        else:
            proj_factorize = False
            proj_dim = int(proj_dim) # Convert to integer

    # Compatibility checking
    if proj_method == "Localize":
        assert args.baseline == "GC", "Localize option only works with GC baseline."
        assert args.layer == "Linear", "Localize option only works with Linear layer."
        assert args.random_drop == 0.0, "Localize option can't be combined with random drop."
        assert proj_factorize, "Localize option only works with factorized projection."

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
        if tda in ["IF-NONE", "IF-RAW", "IF-KFAC", "IF-EKFAC"]:
            train_batch_size = 24
            test_batch_size = 24
    elif baseline == "LoGra":
        if tda in ["IF-NONE", "IF-RAW", "IF-KFAC", "IF-EKFAC"]:
            train_batch_size = 24
            test_batch_size = 24
    elif baseline == "LogIX":
        if tda in ["IF-NONE", "IF-RAW", "IF-KFAC", "IF-EKFAC"]:
            train_batch_size = 32
            test_batch_size = 32
    elif baseline == "dattri":
        if tda in ["TRAK", "GD"]:
            train_batch_size = 4
            test_batch_size = 4
    else:
        raise ValueError("Invalid baseline and tda combination.")

    return train_batch_size, test_batch_size

def result_filename(args):
    filename_parts = []

    if args.projection is not None:
        filename_parts.append(args.projection)


    filename_parts.append(f"thrd-{args.threshold}")
    filename_parts.append(f"rdp-{args.random_drop}")

    training_setting = args.output_dir.split("/")[-1]
    # Join parts and save the file
    result_filename = f"./results/{training_setting}/{args.baseline}/{args.tda}/{args.layer}/{'_'.join(filename_parts)}.pt"

    return result_filename

def lds(score, training_setting):
    def read_nodes(file_path):
        int_list = []
        with open(file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                for item in row:
                    try:
                        int_list.append(int(item))
                    except ValueError:
                        print(f"Warning: '{item}' could not be converted to an integer and was skipped.")
        return int_list

    try:
        score = score.detach().cpu()
        # Prepare node list
        nodes_str = [f"./checkpoints/{training_setting}/{i}/train_index.csv" for i in range(50)]
        full_nodes = list(range(4656))
        node_list = []
        for node_str in nodes_str:
            numbers = read_nodes(node_str)
            index = [full_nodes.index(number) for number in numbers]
            node_list.append(index)

        # Load ground truth
        loss_list = torch.load(f"./results/{training_setting}/gt.pt", map_location=torch.device('cpu')).detach()

        # Calculate approximations
        approx_output = []
        for i in range(len(nodes_str)):
            score_approx_0 = score[node_list[i], :]
            sum_0 = torch.sum(score_approx_0, axis=0)
            approx_output.append(sum_0)

        # Calculate correlations
        res = 0
        counter = 0
        for i in range(score.shape[1]):
            tmp = spearmanr(
                np.array([approx_output[k][i] for k in range(len(approx_output))]),
                np.array([loss_list[k][i].numpy() for k in range(len(loss_list))])
            ).statistic
            if not np.isnan(tmp):
                res += tmp
                counter += 1

        return res/counter if counter > 0 else float('nan'), loss_list, approx_output
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        return None, None, None