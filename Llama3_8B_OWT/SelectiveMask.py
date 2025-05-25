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
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

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
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.46.0")

logger = get_logger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
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

    # >>>>>>>>>>>>>>>>>>>>> Customize Argument begins here >>>>>>>>>>>>>>>>>>>>>
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to be used",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="Linear",
        help="Layer used for attribution.",
    )
    parser.add_argument(
        "--sparsification_dim",
        type=int,
        default=16,
        help="Target number of (factorized) active parameters.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=2000,
        help="Number of epochs for training Selective Mask.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Interval for logging the training process.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of training samples used for training Selective Mask.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=5.0,
        help="Lambda for the regularization term.",
    )
    parser.add_argument(
        "--early_stop",
        type=float,
        default=0.9,
        help="The correlation threshold for early stopping.",
    )
    # >>>>>>>>>>>>>>>>>>>>> Worker parallelization argument >>>>>>>>>>>>>>>>>>>>>
    parser.add_argument(
        "--worker",
        type=str,
        default="0/1",
        help="Worker ID and total workers in format WORKER_ID/TOTAL_WORKER",
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
    send_example_telemetry("run_clm_no_trainer", args)

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
            # torch_dtype=torch.bfloat16, #Add
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
    from Llama3_8B_OWT.utils import SubsetSampler
    from _GradComp.utils.common import find_layers
    from _SelectiveMask.MLPGradientExtractor import MLPGradientExtractor
    from _SelectiveMask.MLPSelectiveMask import MLPSelectiveMask

    if args.device.startswith("cuda"):
        # Check if GPU is available
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please check your CUDA installation.")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
    else:
        assert args.device == "cpu", "Invalid device. Choose from 'cuda' or 'cpu'."
        device = torch.device("cpu")

    # Dataset
    whole_train_dataset = lm_datasets["train"]
    train_batch_size, test_batch_size = 4, 4

    # split the train_dataset into train and test further, with 80% for training and 20% for testing
    train_dataset = whole_train_dataset.select(range(int(args.n * 0.8)))
    test_dataset = whole_train_dataset.select(range(int(args.n * 0.8), args.n))

    train_sampler = SubsetSampler(range(len(train_dataset)))
    train_dataloader = DataLoader(
        train_dataset, collate_fn=default_data_collator, batch_size=train_batch_size, sampler=train_sampler
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=default_data_collator, batch_size=test_batch_size, shuffle=False
    )

    # Extract the layers we want to analyze
    layers = find_layers(model, args.layer, return_type="name_instance")

    # Parse worker configuration
    worker_config = args.worker.split("/")
    if len(worker_config) != 2:
        raise ValueError("Worker argument must be in format 'WORKER_ID/TOTAL_WORKER'")

    worker_id = int(worker_config[0])
    total_workers = int(worker_config[1])

    if worker_id >= total_workers:
        raise ValueError(f"Worker ID {worker_id} must be less than total workers {total_workers}")

    logger.info(f"Running as worker {worker_id} of {total_workers}")

    # Divide layers among workers
    layers_per_worker = (len(layers) + total_workers - 1) // total_workers  # Ceiling division
    start_idx = worker_id * layers_per_worker
    end_idx = min((worker_id + 1) * layers_per_worker, len(layers))

    logger.info(f"Worker {worker_id} processing layers {start_idx} to {end_idx-1} (inclusive)")
    layers_to_process = layers[start_idx:end_idx]

    if not layers_to_process:
        logger.info(f"No layers assigned to worker {worker_id}. Exiting.")
        return

    # Create output directory for saving masks
    output_dir = f"./SelectiveMask/mask_{args.sparsification_dim}*{args.sparsification_dim}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize gradient extractor
    logger.info("Initializing gradient extractor...")
    extractor = MLPGradientExtractor(
        model=model,
        device=device,
        cpu_offload=True,
    )

    # Process each layer individually to save memory
    for layer_idx, (module_name, layer) in enumerate(layers_to_process):
        global_layer_idx = start_idx + layer_idx
        logger.info(f"Processing layer {global_layer_idx + 1}/{len(layers)}: {module_name}")

        # Extract gradients for this specific layer (both train and test)
        train_components, test_components = extractor.extract_gradients_for_layer(
            layer_name=module_name,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader
        )

        if train_components is None or test_components is None:
            logger.warning(f"No gradient components available for layer {module_name}, skipping...")
            continue

        # Extract components
        train_pre_activations = train_components['pre_activation']
        train_input_features = train_components['input_features']
        test_pre_activations = test_components['pre_activation']
        test_input_features = test_components['input_features']
        is_3d = train_components['is_3d']

        # Get dimensions
        pre_activation_dim = train_pre_activations.shape[-1]
        input_features_dim = train_input_features.shape[-1]

        logger.info(f"Layer {global_layer_idx + 1} - Pre-activation dimension: {pre_activation_dim}, "
                   f"Input features dimension: {input_features_dim}")

        logger.info(f"Training the dual component mask optimizer for layer {global_layer_idx + 1}...")

        # Initialize the optimizer for this layer
        optimizer = MLPSelectiveMask(
            pre_activation_dim=pre_activation_dim,
            input_features_dim=input_features_dim,
            lambda_reg=args.regularization,
            lr=args.learning_rate,
            min_active_pre_activation=args.sparsification_dim,
            max_active_pre_activation=args.sparsification_dim,
            min_active_input=args.sparsification_dim,
            max_active_input=args.sparsification_dim,
            device=device,
            logger=logger,
        )

        # Train the optimizer
        eval_metrics = optimizer.train(
            train_pre_activation=train_pre_activations,
            train_input_features=train_input_features,
            test_pre_activation=test_pre_activations,
            test_input_features=test_input_features,
            batch_size=10,
            accumulation_steps=5,
            num_epochs=args.epoch,
            log_every=args.log_interval,
            correlation_threshold=args.early_stop,
            is_3d=is_3d
        )

        # Get and save important indices
        important_indices = optimizer.get_important_indices(
            threshold=0.5,
            min_count_pre=args.sparsification_dim,
            min_count_input=args.sparsification_dim
        )

        # Calculate effective sparsity
        total_params = pre_activation_dim * input_features_dim
        effective_params = len(important_indices['pre_activation']) * len(important_indices['input_features'])
        sparsity = 100 - (effective_params / total_params * 100)

        logger.info(f"Results for Layer {global_layer_idx + 1}:")
        logger.info(f"Pre-activation mask: {len(important_indices['pre_activation'])}/{pre_activation_dim} "
                  f"parameters ({len(important_indices['pre_activation'])/pre_activation_dim*100:.2f}%)")
        logger.info(f"Input features mask: {len(important_indices['input_features'])}/{input_features_dim} "
                  f"parameters ({len(important_indices['input_features'])/input_features_dim*100:.2f}%)")
        logger.info(f"Effective parameters: {effective_params}/{total_params} ({100-sparsity:.2f}%)")
        logger.info(f"Sparsity achieved: {sparsity:.2f}%")
        logger.info(f"Correlation preserved: {eval_metrics['avg_rank_correlation']:.4f}")

        # Save important indices
        output_file = os.path.join(output_dir, f'{module_name}.pt')
        torch.save(important_indices, output_file)

        # Clear memory
        del train_pre_activations, train_input_features, test_pre_activations, test_input_features
        del optimizer, train_components, test_components
        torch.cuda.empty_cache()

    logger.info(f"Worker {worker_id} has completed processing all assigned layers ({start_idx}-{end_idx-1}).")


if __name__ == "__main__":
    main()