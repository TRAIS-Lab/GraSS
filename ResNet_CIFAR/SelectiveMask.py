import argparse
import random
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from _SelectiveMask.SelectiveMask import SelectiveMask
from _SelectiveMask.GradientExtractor import GradientExtractor
from _dattri.benchmark.load import load_benchmark
from _dattri.benchmark.utils import SubsetSampler

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize Selective Mask over gradients for ResNet9 on CIFAR2")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to be used"
    )
    parser.add_argument(
        "--sparsification_dim",
        type=int,
        default=100,
        help="Target number of active parameters across the model."
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=2000,
        help="Number of epochs for training Selective Mask."
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Interval for logging the training process."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of training samples used for training Selective Mask."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=5.0,
        help="Lambda for the regularization term."
    )
    parser.add_argument(
        "--early_stop",
        type=float,
        default=0.9,
        help="The correlation threshold for early stopping."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./SelectiveMask/",
        help="Directory to save the localization results"
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Whether to offload tensors to CPU to save GPU memory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for data loading"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    return args

def setup_logger():
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logger

def create_dataloaders(model_details, args):
    """Create train and test dataloaders for localization"""
    # Get the full training dataset
    train_dataset = model_details["train_dataset"]

    # Create a smaller subset for localization training
    train_indices = list(range(args.n))
    loc_train_sampler = SubsetSampler(train_indices[:int(args.n * 0.8)])
    loc_test_sampler = SubsetSampler(train_indices[int(args.n * 0.8):args.n])

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=loc_train_sampler
    )

    test_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=loc_test_sampler
    )

    return train_dataloader, test_dataloader

def main():
    args = parse_args()

    set_random_seed(args.seed)

    logger = setup_logger()

    # Create output directory
    output_dir = f"{args.output_dir}/mask_{args.sparsification_dim}"
    os.makedirs(output_dir, exist_ok=True)

    # Load ResNet9 + CIFAR2 benchmark
    logger.info(f"Loading ResNet9 + CIFAR2 benchmark...")
    model_details, _ = load_benchmark(model="resnet9", dataset="cifar2", metric="lds")

    # Get model and move to device
    model = model_details["model"]

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please check your CUDA installation.")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
    else:
        assert args.device == "cpu", "Invalid device. Choose from 'cuda' or 'cpu'."
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Calculate total parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {total_params} trainable parameters")

    # Verify localization target is less than total parameters
    if args.sparsification_dim > total_params:
        logger.warning(f"Localization target ({args.sparsification_dim}) exceeds total parameters ({total_params}). Setting to {total_params}.")
        args.sparsification_dim = total_params

    # Create dataloaders for localization
    train_dataloader, test_dataloader = create_dataloaders(model_details, args)

    # Define custom loss function for gradient extraction
    def custom_loss_fn(outputs, batch):
        """Custom loss function for gradient extraction"""
        images, labels = batch
        loss = nn.CrossEntropyLoss()
        loss_value = loss(outputs, labels.to(device))
        return loss_value

    # Initialize gradient extractor
    logger.info("Initializing gradient extractor...")
    extractor = GradientExtractor(
        model=model,
        device=device,
        cpu_offload=args.cpu_offload
    )

    # Extract gradients for the entire model at once
    logger.info("Extracting gradients for the entire model...")
    train_gradients, test_gradients = extractor.extract_gradients(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        custom_loss_fn=custom_loss_fn
    )

    if train_gradients is None or test_gradients is None:
        logger.error("Failed to extract gradients. Exiting.")
        return

    # Get gradient tensors
    train_grad_tensor = train_gradients['gradient']
    test_grad_tensor = test_gradients['gradient']

    # Get gradient dimension (should be total parameters)
    gradient_dim = train_grad_tensor.shape[1]

    logger.info(f"Extracted gradients - Shape: {train_grad_tensor.shape}, Total parameters: {gradient_dim}")

    # Initialize the Selective Mask optimizer for the entire model
    logger.info("Training the gradient mask optimizer...")
    optimizer = SelectiveMask(
        gradient_dim=gradient_dim,
        lambda_reg=args.regularization,
        lr=args.learning_rate,
        min_active_gradient=args.sparsification_dim,
        max_active_gradient=args.sparsification_dim,  # Exact target, no flexibility
        device=device,
        logger=logger
    )

    # Train the Selective Mask
    eval_metrics = optimizer.train(
        train_gradients=train_grad_tensor,
        test_gradients=test_grad_tensor,
        batch_size=4000,
        num_epochs=args.epoch,
        log_every=args.log_interval,
        correlation_threshold=args.early_stop
    )

    # Get important indices
    logger.info(f"Retrieving important gradient indices...")
    important_indices = optimizer.get_important_indices(
        threshold=0.5,
        min_count=args.sparsification_dim
    )

    # Calculate sparsity
    effective_params = len(important_indices['gradient'])
    sparsity = 100 - (effective_params / gradient_dim * 100)

    logger.info(f"Results:")
    logger.info(f"Gradient mask: {effective_params}/{gradient_dim} parameters ({effective_params/gradient_dim*100:.2f}%)")
    logger.info(f"Sparsity achieved: {sparsity:.2f}%")
    logger.info(f"Correlation preserved: {eval_metrics['avg_rank_correlation']:.4f}")

    # Convert to tensor
    active_indices_tensor = torch.tensor(important_indices['gradient'], dtype=torch.long)

    # Save the important indices
    results = {
        "active_indices": active_indices_tensor,
        "total_params": gradient_dim,
        "active_params": effective_params,
        "sparsity": sparsity,
        "correlation": eval_metrics['avg_rank_correlation'],
        "args": vars(args)
    }

    # Get parameter mapping (for analysis)
    param_map = extractor.get_param_to_indices_map()
    results["param_map"] = param_map

    # Analyze which parameters were selected
    logger.info("\nParameter-wise analysis:")
    param_stats = {}
    for name, (start_idx, end_idx) in param_map.items():
        # Count how many indices in this parameter range are active
        param_active_indices = [idx for idx in important_indices['gradient'] if start_idx <= idx < end_idx]
        param_active_count = len(param_active_indices)
        param_total = end_idx - start_idx
        param_sparsity = 100 - (param_active_count / param_total * 100) if param_total > 0 else 0

        param_stats[name] = {
            "active": param_active_count,
            "total": param_total,
            "sparsity": param_sparsity,
            "percentage": param_active_count / effective_params * 100
        }

        if param_active_count > 0:
            logger.info(f"Parameter {name}: {param_active_count}/{param_total} active "
                      f"({param_active_count/param_total*100:.2f}% dense, "
                      f"{param_active_count/effective_params*100:.2f}% of all active)")

    # Add parameter stats to results
    results["param_stats"] = param_stats

    # Save results
    output_file = os.path.join(output_dir, f'result_{args.seed}.pt')
    torch.save(results, output_file)
    logger.info(f"Results saved to {output_file}")

    # Clear memory
    del train_grad_tensor, test_grad_tensor, optimizer
    del train_gradients, test_gradients
    torch.cuda.empty_cache()

    logger.info("Localization completed successfully!")

if __name__ == "__main__":
    main()