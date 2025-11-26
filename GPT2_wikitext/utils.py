from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator, List

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

import numpy as np
from scipy.stats import spearmanr
import csv

# Import shared utilities from compressor
from _dattri.algorithm.block_projected_if.core.compressor import setup_compression_kwargs

# Import SubsetSampler from dattri (canonical location)
from _dattri.benchmark.utils import SubsetSampler

# Re-export for backward compatibility
__all__ = ['replace_conv1d_modules', 'SubsetSampler', 'setup_compression_kwargs', 'result_filename', 'lds', 'split_lds']


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

def result_filename(args):
    """Generate result filename based on experiment arguments.

    Note: sparsification and projection are required arguments.
    """
    training_setting = args.output_dir.split("/")[-1]
    return f"./results/{training_setting}/{args.baseline}/{args.hessian}/{args.layer}/{args.sparsification}->{args.projection}.pt"

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

def split_lds(score, training_setting, split_indices, full_test_len):
    """
    Calculate LDS on a subset of the test data.

    Args:
        score: Attribution scores (train_size x test_subset_size)
        training_setting: Model setting for loading ground truth
        split_indices: Indices of the test subset in the full test set
        full_test_len: Length of the full test set

    Returns:
        lds_score: LDS score for this subset
    """
    # Create a full score tensor with zeros for test examples not in the split
    full_score = torch.zeros((score.shape[0], full_test_len), dtype=score.dtype, device=score.device)

    # Place the split scores at the correct positions
    for i, idx in enumerate(split_indices):
        full_score[:, idx] = score[:, i]

    # Calculate LDS using the original function but with our prepared full_score
    lds_result, _, _ = lds(full_score, training_setting)

    return lds_result