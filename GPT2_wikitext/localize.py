#!/usr/bin/env python
# coding=utf-8
"""
Parameter localization for causal language models.
This simplified script focuses only on the localization aspect.
"""

import argparse
import json
import logging
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import copy

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

logger = get_logger(__name__)

class Localizer(nn.Module):
    """Parameter localization for language models"""

    def __init__(self, model, pretrained_model, finetuned_model, args):
        super(Localizer, self).__init__()

        self.model = model
        self.pretrained_model = pretrained_model
        self.finetuned_model = finetuned_model
        self.args = args

        self.device = model.device
        self.pretrained_model.to(self.device)
        self.finetuned_model.to(self.device)

        # Set models to eval mode
        self.model.eval()
        self.pretrained_model.eval()
        self.finetuned_model.eval()

        # Freeze pretrained and finetuned models
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.finetuned_model.parameters():
            param.requires_grad = False

        # Initialize trainable parameters
        self.trainable_params = self._get_trainable_params()

        # Create binary masks
        self.create_binary_masks()

    def _get_trainable_params(self):
        """Get dictionary of trainable parameters from transformer blocks"""
        params = {}
        for n, p in self.model.named_parameters():
            # Only select parameters from transformer blocks (not embeddings or lm_head)
            if 'transformer.h' in n:
                params[n] = p
        return params

    def create_binary_masks(self):
        """Initialize mask parameters and task vectors"""
        self.trainable_name = []
        self.trainable_parameters = []

        # Register trainable parameters
        for n in self.trainable_params:
            self.trainable_name.append(n)
            p = self.trainable_params[n]
            self.trainable_parameters.append(torch.rand_like(p.data, device=self.device, requires_grad=False))

        self.num_params = sum([p.numel() for p in self.trainable_parameters])

        # Create task vectors (difference between finetuned and pretrained)
        self.task_vectors = []
        for name in self.trainable_name:
            pretensor = None
            finetensor = None

            for pre_n, pre_p in self.pretrained_model.named_parameters():
                if pre_n == name:
                    pretensor = pre_p.to(self.device)

            for fine_n, fine_p in self.finetuned_model.named_parameters():
                if fine_n == name:
                    finetensor = fine_p.to(self.device)

            if pretensor is not None and finetensor is not None:
                self.task_vectors.append((finetensor - pretensor).detach())
            else:
                # Handle case where parameter name doesn't match
                logger.warning(f"Parameter {name} not found in both models")
                self.task_vectors.append(torch.zeros_like(self.trainable_params[name]))

        # Initialize mask based on top parameter differences
        self._create_base_mask()

    def _create_base_mask(self):
        """Create initial mask based on largest parameter differences"""
        sparsity = self.args.localize_sparsity
        sigmoid_bias = self.args.sigmoid_bias

        # Flatten all task vectors
        abs_tv = []
        for p in self.task_vectors:
            abs_tv.append(torch.abs(p).view(-1))

        abs_tv = torch.cat(abs_tv)
        k = int(sparsity * abs_tv.numel())  # Top k% of parameters

        # Get threshold value
        values, _ = torch.topk(abs_tv.view(-1), k)
        threshold = values.min()

        # Create binary masks using sigmoid bias
        self.mask = []
        for p in self.task_vectors:
            mask = torch.zeros_like(p, requires_grad=True)
            mask.data[torch.absolute(p) > threshold] = sigmoid_bias
            mask.data[torch.absolute(p) <= threshold] = -sigmoid_bias
            self.mask.append(mask)

        # Log how many parameters will be modified
        masked_params = sum([torch.sum(torch.nn.Sigmoid()(p) > 0.5) for p in self.mask])
        logger.info(f"Initial mask selects {masked_params / self.num_params:.2%} of parameters")

    def reset_model(self):
        """Reset model to pretrained weights"""
        for i, name in enumerate(self.trainable_name):
            pretensor = None

            for pre_n, pre_p in self.pretrained_model.named_parameters():
                if pre_n == name:
                    pretensor = pre_p.to(self.device)

            if pretensor is not None:
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        if n == name:
                            p.copy_(pretensor)

    def apply_mask(self, return_mask=False, round_values=True):
        """Apply binary mask to model parameters"""
        sigmoid = torch.nn.Sigmoid()
        n_masked_params = 0
        binary_mask = []

        for i, name in enumerate(self.trainable_name):
            pretensor = None
            finetensor = None

            for pre_n, pre_p in self.pretrained_model.named_parameters():
                if pre_n == name:
                    pretensor = pre_p.to(self.device)

            for fine_n, fine_p in self.finetuned_model.named_parameters():
                if fine_n == name:
                    finetensor = fine_p.to(self.device)

            if pretensor is not None and finetensor is not None:
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        if n == name:
                            # Calculate mask values
                            mask_values = sigmoid(self.mask[i])

                            # Round to binary values if requested
                            if round_values:
                                mask_values = torch.round(mask_values)
                                binary_mask.append(mask_values)

                            # Count masked parameters
                            n_masked_params += torch.sum(mask_values > 0.5)

                            # Apply mask
                            update = mask_values * (finetensor - pretensor)
                            p.add_(update)

        # Report percentage of parameters being modified
        masked_percent = n_masked_params.item() / self.num_params
        logger.info(f"Masked parameters: {masked_percent:.2%}")

        if return_mask:
            return binary_mask, masked_percent

    def train_mask(self, dataloader, num_epochs):
        """Train the mask using gradients from loss function"""
        sigmoid = torch.nn.Sigmoid()
        loss_log = []

        # Optimizer for mask parameters
        optimizer = torch.optim.Adam(self.mask, lr=self.args.localize_lr)

        for epoch in range(num_epochs):
            # Reset model to pretrained weights
            self.reset_model()

            # Apply current mask (non-binary)
            self.apply_mask(round_values=False)

            # Training loop
            total_loss = 0
            batch_count = 0

            # Process batches
            for batch in tqdm(dataloader, desc=f"Training mask (epoch {epoch+1}/{num_epochs})"):
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                total_loss += loss.item()
                batch_count += 1

                # Update masks
                optimizer.step()
                optimizer.zero_grad()

                # Apply L1 regularization to encourage sparsity
                with torch.no_grad():
                    for mask_param in self.mask:
                        mask_vals = sigmoid(mask_param)
                        # L1 penalty pushing values toward 0 or 1
                        l1_grad = self.args.l1_strength * (2 * mask_vals - 1)
                        mask_param.data.add_(-self.args.localize_lr * l1_grad)

            # Average loss for epoch
            avg_loss = total_loss / batch_count
            loss_log.append(avg_loss)

            # Evaluate and log current mask
            self.reset_model()
            binary_mask, mask_percent = self.apply_mask(return_mask=True, round_values=True)

            # Evaluate on a few batches
            eval_loss = 0.0
            eval_count = 0
            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= 5:  # Evaluate on 5 batches
                        break
                    outputs = self.model(**batch)
                    eval_loss += outputs.loss.item()
                    eval_count += 1

            avg_eval_loss = eval_loss / eval_count
            logger.info(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, eval_loss={avg_eval_loss:.4f}, mask_percent={mask_percent:.2%}")

            # Reset for next epoch
            self.model.train()
            self.reset_model()

        # Final mask application
        self.reset_model()
        binary_mask, mask_percent = self.apply_mask(return_mask=True, round_values=True)

        return binary_mask, mask_percent, loss_log

def parse_args():
    parser = argparse.ArgumentParser(description="Parameter localization for CLM")
    # Model and dataset args
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model")
    parser.add_argument("--finetuned_model_path", type=str, help="Path to previously fine-tuned model to analyze")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./localized_model")
    parser.add_argument("--seed", type=int, default=42)

    # Localization-specific args
    parser.add_argument("--localize_epochs", type=int, default=10)
    parser.add_argument("--localize_lr", type=float, default=1e-3)
    parser.add_argument("--localize_sparsity", type=float, default=0.01)
    parser.add_argument("--l1_strength", type=float, default=0.1)
    parser.add_argument("--sigmoid_bias", type=float, default=5.0)
    parser.add_argument("--use_gradient_training", action="store_true",
                       help="Whether to train mask with gradients or just use magnitude-based selection")
    parser.add_argument("--eval_steps", type=int, default=100)

    return parser.parse_args()

def prepare_dataset(tokenizer, args):
    """Load and prepare the dataset for mask training/evaluation"""
    # Load dataset
    if args.dataset_name is not None:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:5%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[5%:15%]",  # We use a small subset for localization
            )
    else:
        raise ValueError("Please provide a dataset name")

    # Preprocessing function
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

    # Group texts into chunks
    block_size = args.block_size

    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Drop remainder
        total_length = (total_length // block_size) * block_size
        # Split by chunks
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_batch_size
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=args.per_device_batch_size
    )

    return train_dataloader, eval_dataloader

def localize_parameters():
    """Main function for parameter localization"""
    args = parse_args()

    # Set up accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        mask_dir = os.path.join(args.output_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)

    # Load tokenizer
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load datasets
    train_dataloader, eval_dataloader = prepare_dataset(tokenizer, args)

    # Load models
    logger.info(f"Loading pretrained model from {args.model_name_or_path}")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)

    logger.info(f"Loading finetuned model from {args.finetuned_model_path}")
    finetuned_model = AutoModelForCausalLM.from_pretrained(args.finetuned_model_path, config=config)

    # Create a copy of pretrained model for localization
    model = copy.deepcopy(pretrained_model)

    # Prepare with accelerator
    model, train_dataloader = accelerator.prepare(model, train_dataloader)
    pretrained_model = pretrained_model.to(device)
    finetuned_model = finetuned_model.to(device)

    # Initialize localizer
    localizer = Localizer(model, pretrained_model, finetuned_model, args)

    # Train the mask or use static magnitude-based mask
    if args.use_gradient_training:
        logger.info("Training mask using gradients...")
        binary_mask, mask_percent, loss_log = localizer.train_mask(
            train_dataloader,
            args.localize_epochs
        )
    else:
        logger.info("Using static magnitude-based mask...")
        localizer.reset_model()
        binary_mask, mask_percent = localizer.apply_mask(return_mask=True, round_values=True)
        loss_log = []

    # Save the mask
    if accelerator.is_main_process:
        mask_filename = f"mask_sparsity_{args.localize_sparsity:.3f}.pt"
        mask_path = os.path.join(args.output_dir, "masks", mask_filename)

        mask_dict = {
            "binary_mask": binary_mask,
            "trainable_names": localizer.trainable_name,
            "sparsity": mask_percent,
            "args": vars(args)
        }

        torch.save(mask_dict, mask_path)
        logger.info(f"Saved mask to {mask_path}")

        # Save loss log if available
        if loss_log:
            with open(os.path.join(args.output_dir, "masks", f"loss_log_{args.localize_sparsity:.3f}.json"), 'w') as f:
                json.dump({"loss": loss_log}, f)

    # Save the localized model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save configuration
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "localization_config.json"), "w") as f:
            json.dump({
                "localize_sparsity": args.localize_sparsity,
                "localize_lr": args.localize_lr,
                "l1_strength": args.l1_strength,
                "sigmoid_bias": args.sigmoid_bias,
                "mask_percent": mask_percent,
                "total_params": localizer.num_params,
                "masked_params": int(mask_percent * localizer.num_params)
            }, f, indent=2)

    logger.info("Localization complete!")

if __name__ == "__main__":
    localize_parameters()