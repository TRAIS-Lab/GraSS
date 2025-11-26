"""
MusicTransformer + MAESTRO experiment for TRAK attribution.

Usage:
    python score.py --proj_type normal --proj_dim 1024
    python score.py --proj_type sjlt --proj_dim 4096

Projection types: normal, rademacher, sjlt, random_mask, grass, identity
"""

import argparse
import os
import sys
import numpy as np

# Add parent directory to path for _dattri import
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import torch
from torch import nn

from _dattri.algorithm.trak import TRAKAttributor
from _dattri.benchmark.load import load_benchmark
from _dattri.benchmark.utils import SubsetSampler
from _dattri.metric import lds
from _dattri.task import AttributionTask
from _dattri.benchmark.models.MusicTransformer.utilities.constants import TOKEN_PAD

def create_validation_split(test_dataset, test_sampler, groundtruth, val_ratio=0.1, seed=42):
    """
    Split the test set into validation and test sets, and reconstruct groundtruth.

    Args:
        test_dataset: The test dataset
        test_sampler: The test sampler
        groundtruth: Tuple of (ground_truth_values, subset_indices)
        val_ratio: Ratio of test data to use for validation
        seed: Random seed for reproducibility

    Returns:
        val_dataloader: DataLoader for validation set
        test_dataloader: DataLoader for test set
        val_gt: Groundtruth for validation set
        test_gt: Groundtruth for test set
    """
    # Get test indices from the sampler
    test_indices = list(test_sampler)
    num_test = len(test_indices)

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Shuffle indices and split
    np.random.shuffle(test_indices)
    num_val = int(val_ratio * num_test)
    val_indices = test_indices[:num_val]
    new_test_indices = test_indices[num_val:]

    # Create validation and test samplers
    val_sampler = SubsetSampler(val_indices)
    new_test_sampler = SubsetSampler(new_test_indices)

    # Create dataloaders
    val_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        sampler=val_sampler,
    )

    new_test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        sampler=new_test_sampler,
    )

    # Reconstruct groundtruth
    original_gt_values, subset_indices = groundtruth

    # Map original test indices to positions in the groundtruth tensor
    test_indices_dict = {idx: pos for pos, idx in enumerate(test_sampler)}

    # Extract validation groundtruth
    val_gt_indices = [test_indices_dict[idx] for idx in val_indices]
    val_gt_values = original_gt_values[:, val_gt_indices]

    # Extract test groundtruth
    test_gt_indices = [test_indices_dict[idx] for idx in new_test_indices]
    test_gt_values = original_gt_values[:, test_gt_indices]

    # Return validation and test sets with reconstructed groundtruth
    return (
        val_dataloader,
        new_test_dataloader,
        (val_gt_values, subset_indices),
        (test_gt_values, subset_indices)
    )

def main():
    parser = argparse.ArgumentParser(description="TRAK attribution for MusicTransformer on MAESTRO")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--proj_type", type=str, default="normal",
                        help="Projection type: normal, rademacher, sjlt, fjlt, random_mask, selective_mask, grass, grass_N, selective_grass, selective_grass_N, identity")
    parser.add_argument("--proj_dim", type=int, default=1024, help="Projection dimension")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of test data to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Define the grid of damping values to search
    damping_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    # Print the settings
    print("Settings: MusicTransformer + MAESTRO")
    print("Projection Type:", args.proj_type)
    print("Projection Dimension:", args.proj_dim)
    print("Validation Split Ratio:", args.val_ratio)
    print("Random Seed:", args.seed)
    print("Damping Grid Search Values:", damping_values)

    # create cifar 10 dataset
    model_details, groundtruth = load_benchmark(model="musictransformer", dataset="maestro", metric="lds")

    print("Original groundtruth shapes:")
    print("Ground truth values shape:", groundtruth[0].shape)
    print("Subset indices shape:", groundtruth[1].shape)

    model = model_details["model"].to(args.device)
    model = model.eval()

    def loss_trak(params, data_target_pair):
        x, y = data_target_pair
        x_t = x.unsqueeze(0)
        y_t = y.unsqueeze(0)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, reduction='none')

        output = torch.func.functional_call(model_details["model"], params, x_t)
        output_last = output[:, -1, :]
        y_last = y_t[:, -1]

        logp = -loss_fn(output_last, y_last)
        logit_func = logp - torch.log(1 - torch.exp(logp))
        return logit_func.squeeze(0)


    def correctness_p(params, data_target_pair):
        x, y = data_target_pair
        x_t = x.unsqueeze(0)
        y_t = y.unsqueeze(0)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, reduction='none')

        output = torch.func.functional_call(model_details["model"], params, x_t)
        output_last = output[:, -1, :]
        y_last = y_t[:, -1]
        logp = -loss_fn(output_last, y_last)

        return torch.exp(logp)

    train_dataloader = torch.utils.data.DataLoader(
        model_details["train_dataset"],
        batch_size=16,
        shuffle=False,
        sampler=model_details["train_sampler"],
    )

    # Split test data into validation and test sets
    val_dataloader, test_dataloader, val_gt, test_gt = create_validation_split(
        model_details["test_dataset"],
        SubsetSampler(range(len(model_details["test_dataset"]))),
        groundtruth,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    print("\nAfter splitting:")
    print("Validation groundtruth values shape:", val_gt[0].shape)
    print("Test groundtruth values shape:", test_gt[0].shape)

    task = AttributionTask(model=model, loss_func=loss_trak, checkpoints=model_details["models_half"][:5])

    # Setup projector kwargs
    projector_kwargs = {
        "device": args.device,
        "proj_type": args.proj_type,
        "proj_seed": args.seed,
        "proj_dim": args.proj_dim,
        "proj_max_batch_size": 64,
    }

    # Grid search over damping values
    best_damping = None
    best_lds_score = float('-inf')
    validation_results = {}

    # Prepare attributor and cache once (reused for different damping values)
    base_attributor = TRAKAttributor(
        task=task,
        correct_probability_func=correctness_p,
        device=args.device,
        projector_kwargs=projector_kwargs,
    )

    # Grid search through damping values
    print("\nPerforming grid search over damping values:")
    for damping in damping_values:
        print(f"\nEvaluating damping = {damping}")

        # Update the regularization parameter
        base_attributor.regularization = damping

        base_attributor.cache(train_dataloader)
        # Evaluate on validation set
        with torch.no_grad():
            val_score = base_attributor.attribute(test_dataloader=val_dataloader)

        val_lds_score = lds(val_score, val_gt)[0]
        mean_val_lds = torch.mean(val_lds_score[~torch.isnan(val_lds_score)]).item()
        validation_results[damping] = mean_val_lds

        print(f"Damping: {damping}, Validation LDS: {mean_val_lds}")

        # Track best damping value
        if mean_val_lds > best_lds_score:
            best_lds_score = mean_val_lds
            best_damping = damping

    print("\nValidation Results:")
    for damping, score in validation_results.items():
        print(f"Damping: {damping}, LDS: {score}")

    print(f"\nBest damping value: {best_damping} (Validation LDS: {best_lds_score})")

    # Evaluate the best damping value on the test set
    print("\nEvaluating best damping value on test set...")
    base_attributor.regularization = best_damping

    proj_time = base_attributor.cache(train_dataloader)
    with torch.no_grad():
        test_score = base_attributor.attribute(test_dataloader=test_dataloader)

    test_lds_score = lds(test_score, test_gt)[0]
    mean_test_lds = torch.mean(test_lds_score[~torch.isnan(test_lds_score)]).item()

    print(f"Final Results:")
    print(f"Best Damping: {best_damping}")
    print(f"Validation LDS: {best_lds_score}")
    print(f"Test LDS: {mean_test_lds}")

    result = {
        "best_damping": best_damping,
        "lds": mean_test_lds,
        "proj_time": proj_time,
    }

    os.makedirs("./results", exist_ok=True)
    torch.save(result, f"./results/{args.proj_type}-{args.proj_dim}.pt")

if __name__ == "__main__":
    main()