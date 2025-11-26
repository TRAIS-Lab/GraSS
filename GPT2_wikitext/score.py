#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import csv
from pathlib import PosixPath
import time

import sys

# Add parent directory to path for _dattri import
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.46.0")

logger = get_logger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def create_validation_split(test_dataset, val_ratio=0.1, seed=42):
    """
    Split the test set into validation and test sets.

    Args:
        test_dataset: The test dataset
        val_ratio: Ratio of test data to use for validation
        seed: Random seed for reproducibility

    Returns:
        val_dataset: Dataset for validation
        test_dataset: Dataset for test
        val_indices: Indices for validation set
        test_indices: Indices for test set
    """
    # Get total number of examples
    num_test = len(test_dataset)
    all_indices = list(range(num_test))

    # Set random seed for reproducibility
    random.seed(seed)

    # Shuffle indices and split
    random.shuffle(all_indices)
    num_val = int(val_ratio * num_test)
    val_indices = all_indices[:num_val]
    test_indices = all_indices[num_val:]

    # Create validation and test datasets
    val_dataset = test_dataset.select(val_indices)
    new_test_dataset = test_dataset.select(test_indices)

    return val_dataset, new_test_dataset, val_indices, test_indices

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--subset_ratio",
        type=float,
        default=1.0,
        help="The ratio used for model training.",
    )

    # >>>>>>>>>>>>>>>>>>>>> Customize Argument begins here >>>>>>>>>>>>>>>>>>>>>
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to be used",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Record profiling results.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="dattri",
        help="Specify which baseline library implementation we want to run the data attribution method. Available options: dattri, LogIX.",
    )
    parser.add_argument(
        "--hessian",
        type=str,
        default="eFIM",
        choices=["eFIM", "ekfac", "Identity"],
        help="Hessian approximation type. eFIM: empirical Fisher, ekfac: EK-FAC, Identity: no Hessian.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="Linear",
        help="Layer used for attribution.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default="identity",
        help="Projection (Stage 2): non-factorized compression. Format: 'TYPE-DIM' (e.g., 'sjlt-4096'). Types: normal, rademacher, sjlt, fjlt, random_mask, selective_mask, grass, grass_N, selective_grass, selective_grass_N, identity.",
    )
    parser.add_argument(
        "--sparsification",
        type=str,
        default="identity",
        help="Sparsification (Stage 1): factorized compression. Format: 'TYPE-DIM*DIM' (e.g., 'random_mask-128*128'). Types: normal, rademacher, sjlt, fjlt, random_mask, grass, grass_N, selective_grass, selective_grass_N, identity.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of test data to use for validation",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name, trust_remote_code=args.trust_remote_code
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                trust_remote_code=args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                trust_remote_code=args.trust_remote_code,
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    # >>>>>>>>>>>>>>>>>>>>> Customized Code begins here >>>>>>>>>>>>>>>>>>>>>
    from GPT2_wikitext.utils import SubsetSampler, replace_conv1d_modules, setup_compression_kwargs, result_filename, split_lds

    if args.device.startswith("cuda"):
        # Check if GPU is available
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please check your CUDA installation.")
        device = torch.device(args.device)
    else:
        assert args.device == "cpu", "Invalid device. Choose from 'cuda' or 'cpu'."
        device = torch.device("cpu")

    torch.cuda.set_device(device)

    # Dataset
    train_dataset = lm_datasets["train"]
    test_dataset = lm_datasets["validation"]
    train_batch_size, test_batch_size = 32, 32

    # Split test dataset into validation and test
    val_dataset, new_test_dataset, val_indices, test_indices = create_validation_split(
        test_dataset, val_ratio=args.val_ratio, seed=args.seed
    )

    # Create dataloaders
    train_sampler = SubsetSampler(range(len(train_dataset)))
    train_dataloader = DataLoader(
        train_dataset, collate_fn=default_data_collator, batch_size=train_batch_size, sampler=train_sampler
    )
    val_dataloader = DataLoader(
        val_dataset, collate_fn=default_data_collator, batch_size=test_batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        new_test_dataset, collate_fn=default_data_collator, batch_size=test_batch_size, shuffle=False
    )
    # Save the original test dataset length for proper LDS calculation
    original_test_len = len(test_dataset)

    training_setting = args.output_dir.split("/")[-1]

    # Define the grid of damping values to search
    damping_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    best_damping = None
    best_lds_score = float('-inf')
    validation_results = {}
    throughput_stats = {}

    sparsifier_kwargs, projector_kwargs = setup_compression_kwargs(args, device)

    # Logging setting
    logger.info(f"The train dataset length: {len(train_dataset)}.")
    logger.info(f"The test dataset length: {len(test_dataset)}.")
    logger.info(f"The train batch size: {train_batch_size}")
    logger.info(f"The test batch size: {test_batch_size}")
    logger.info(f"TDA Method: {args.baseline}, Hessian: {args.hessian}")
    logger.info(f"Sparsifier: {sparsifier_kwargs}")
    logger.info(f"Projector: {projector_kwargs}")
    logger.info(f"Layer: {args.layer}")
    logger.info("***** Running attribution *****")

    profile = None
    if args.baseline == "dattri":
        from _dattri.algorithm import BlockProjectedIFAttributor
        from _dattri.task import AttributionTask
        import torch.nn as nn

        # Use hessian type directly (already in dattri naming convention)

        model_id = 0
        checkpoint = f"{args.output_dir}/{model_id}"

        # Define loss function for dattri
        def loss_func(model_inst, batch, dev):
            inputs = {k: v.to(dev) for k, v in batch.items()}
            outputs = model_inst(**inputs)
            return outputs.loss

        def m(model_inst, batch, dev):
            inputs = {k: v.to(dev) for k, v in batch.items()}
            outputs = model_inst(**inputs)
            logp = -outputs.loss
            return logp - torch.log(1 - torch.exp(logp))

        # Define checkpoint loader
        def checkpoints_load_func(model_instance, checkpoint_path):
            model_instance = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
            model_instance.eval()
            return replace_conv1d_modules(model_instance)

        # Load model and find layer names
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        model = replace_conv1d_modules(model)
        layer_names = [
            name for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ] if args.layer == "Linear" else None

        # Create attribution task
        task = AttributionTask(
            model=model,
            loss_func=loss_func,
            checkpoints=checkpoint,
            target_func=m,
            checkpoints_load_func=checkpoints_load_func,
        )

        attributor = BlockProjectedIFAttributor(
            task=task,
            layer_names=layer_names,
            hessian=args.hessian,
            damping=None,
            device=device,
            sparsifier_kwargs=sparsifier_kwargs,
            projector_kwargs=projector_kwargs,
            offload="disk", # for comparison with LogIX, can be safely changed to "cpu"
            cache_dir=f"./dattri/{args.sparsification}->{args.projection}"
        )

        # Measure cache throughput
        torch.cuda.synchronize(device)
        cache_start_time = time.time()
        attributor.cache(train_dataloader)
        torch.cuda.synchronize(device)
        cache_end_time = time.time()

        # Grid search over damping values
        logger.info("Starting grid search for damping values...")
        for damping in tqdm(damping_values, desc="Damping Grid Search"):
            logger.info(f"Evaluating damping = {damping}")

            # Compute preconditioners for current damping
            attributor.damping = damping
            attributor.compute_preconditioners(damping=damping)
            attributor.compute_ifvp()

            # Evaluate on validation set
            val_score = attributor.attribute(train_dataloader, val_dataloader)
            # Calculate LDS for validation set
            val_lds_score = split_lds(val_score, training_setting, val_indices, original_test_len)
            validation_results[damping] = val_lds_score

            logger.info(f"Damping: {damping}, Validation LDS: {val_lds_score}")

            # Track best damping value
            if val_lds_score > best_lds_score:
                best_lds_score = val_lds_score
                best_damping = damping

        logger.info("\nValidation Results:")
        for damping, score in validation_results.items():
            logger.info(f"Damping: {damping}, LDS: {score}")

        logger.info(f"\nBest damping value: {best_damping} (Validation LDS: {best_lds_score})")

        # Run final attribution with best damping value
        logger.info("\nRunning final attribution with best damping value...")
        torch.cuda.synchronize(device)
        attribute_start_time = time.time()

        # Compute preconditioners for best damping value
        attributor.damping = best_damping
        attributor.compute_preconditioners(damping=best_damping)
        attributor.compute_ifvp()

        # Measure attribute throughput
        torch.cuda.synchronize(device)
        attribute_start_time = time.time()
        score = attributor.attribute(train_dataloader, test_dataloader)
        torch.cuda.synchronize(device)
        attribute_end_time = time.time()

    elif args.baseline == "LogIX":
        from _LogIX.huggingface import LogIXArguments, patch_trainer

        # Map dattri hessian naming to LogIX naming
        logix_hessian_mapping = {"eFIM": "raw", "ekfac": "ekfac", "Identity": "none"}
        logix_hessian = logix_hessian_mapping.get(args.hessian, "none")
        assert args.layer == "Linear", "LogIX only supports Linear setting now."
        assert args.sparsification != "identity", "LogIX requires sparsification method."

        LogIXTrainer = patch_trainer(transformers.Trainer)

        # 1. Computing EK-FAC factors for training data
        model_id = 0
        checkpoint = f"{args.output_dir}/{model_id}"
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        model = replace_conv1d_modules(model)
        model = model.to(device)
        model.eval()

        logix_args_train = LogIXArguments(
            project=f"./LogIX/{args.sparsification}",
            config=f"./LogIX/{args.sparsification}.yaml",
            lora=True,
            hessian=logix_hessian,
            save="grad",
            train_data=True,
            label_key="input_ids",
        )
        training_args = transformers.TrainingArguments(
            output_dir=f"./LogIX/",
            num_train_epochs=1,
            per_device_train_batch_size=train_batch_size,
            report_to="none",
        )
        trainer = LogIXTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            data_collator=default_data_collator,
            args=training_args,
            logix_args=logix_args_train,
        )

        # Measure cache throughput
        torch.cuda.synchronize(device)
        cache_start_time = time.time()
        trainer.extract_log()
        torch.cuda.synchronize(device)
        cache_end_time = time.time()

        # 2. Computing influence scores for test data
        model = AutoModelForCausalLM.from_pretrained(checkpoint) # reinitialize the model
        model = replace_conv1d_modules(model)
        model = model.to(device)
        model.eval()
        logix_args_test = LogIXArguments(
            project=f"./LogIX/{args.sparsification}",
            config=f"./LogIX/{args.sparsification}.yaml",
            lora=True,
            hessian=logix_hessian,
            save="grad",
            train_data=False,
            label_key="input_ids",
            initialize_from_log=True,
            log_batch_size=32,
        )
        training_args = transformers.TrainingArguments(
            output_dir=f"./LogIX/",
            num_train_epochs=1,
            per_device_train_batch_size=test_batch_size,
            report_to="none",
            gradient_accumulation_steps=1,
        )
        trainer = LogIXTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=new_test_dataset,
            data_collator=default_data_collator,
            args=training_args,
            logix_args=logix_args_test,
        )

        # Measure attribute throughput
        torch.cuda.synchronize(device)
        attribute_start_time = time.time()
        result = trainer.influence()
        torch.cuda.synchronize(device)
        attribute_end_time = time.time()

        score = result["influence"].T

    else:
        raise ValueError("Invalid baseline implementation method. Choose from 'dattri', 'LogIX'.")

    # Calculate throughput
    train_tokens = block_size * len(train_dataset)
    train_test_pairs = len(train_dataset) * len(test_dataset)

    cache_duration = cache_end_time - cache_start_time
    cache_throughput = train_tokens / cache_duration
    throughput_stats["cache"] = {
        "tokens": train_tokens,
        "duration_seconds": cache_duration,
        "throughput_tokens_per_second": cache_throughput
    }

    attribute_duration = attribute_end_time - attribute_start_time
    attribute_throughput = train_test_pairs / attribute_duration
    throughput_stats["attribute"] = {
        "train_test_pairs": train_test_pairs,
        "duration_seconds": attribute_duration,
        "throughput_pair_per_second": attribute_throughput
    }

    lds_score = split_lds(score, training_setting, test_indices, original_test_len)

    logger.info("***** Attribution finished *****")

    result = {"score": score, "lds": lds_score, "profile": profile, "throughput": throughput_stats, "best_damping": best_damping}
    logger.info(result)

    result_path = result_filename(args)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    torch.save(result, result_path)

if __name__ == "__main__":
    main()
