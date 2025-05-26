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

# os.environ["TOKENIZERS_PARALLELISM"] = "1"

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


# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )

def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not exporting memory snapshot")
       return
   file_prefix = f"memory_snapshot_{time.strftime('%Y%m%d_%H%M%S')}"

   try:
       logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
   except Exception as e:
       logger.error(f"Failed to capture memory snapshot {e}")
       return

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
        default="GC",
        help="Specify which baseline library implementation we want to run the data attribution method. Available options: GC, LoGra, dattri.",
    )
    parser.add_argument(
        "--tda",
        type=str,
        default="IF-GC",
        help="Specify which mode we want to run the data attribution method. Available options: IF-{GC,LoGra}, GD-{IF,dattri}, TRAK-{dattri}.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="Linear",
        help="Layer used for attribution.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache/",
        help="Directory to store cache files"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default=None,
        help="The projection method to be used when attributing. Basic format: 'proj_method-proj_dim' for non-factorized gradient and 'proj_method-proj_dim*proj_dim' for factorized gradient.",
    )
    parser.add_argument(
        "--sparsification",
        type=str,
        default=None,
        help="The first stage of the gradient compression algorithm. Basic format: ''sparsification_method-proj_dim' for non-factorized gradient and 'sparsification_method-proj_dim*proj_dim' for factorized gradient.",
    )
    parser.add_argument(
        "--worker",
        type=str,
        default="0/1",
        help="The setup of the worker: format: {worker_id}/{total_workers}.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cache",
        help="The mode of the computation. Available options: 'cache', 'precondition', 'ifvp', 'self_influence', 'attribute', and 'quant'.",
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
        if args.baseline == "GC":
            config = AutoConfig.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=args.trust_remote_code,
            )
        elif args.baseline == "LogIX":
            import json
            from pathlib import Path
            from transformers.models.llama.configuration_llama import LlamaConfig

            # Path to your original config
            model_name = "meta-llama/Llama-3.1-8B-Instruct"
            cache_dir = "/work/10367/pbb/vista/.cache/hub"

            # Get the config path directly
            snapshot_path = Path(cache_dir) / "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
            config_path = snapshot_path / "config.json"

            # Load and modify the config
            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            # Fix the rope_scaling structure
            if "rope_scaling" in config_dict:
                factor = config_dict["rope_scaling"].get("factor", 1.0)
                config_dict["rope_scaling"] = {
                    "type": "linear",
                    "factor": factor
                }

            # Create a LlamaConfig directly instead of using AutoConfig.from_dict
            config = LlamaConfig(**config_dict)
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
            torch_dtype=torch.bfloat16, #Add
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
    from Llama3_8B_OWT.utils import SubsetSampler, FilePromptDataset, prompt_collate_fn, generate_responses, setup_compression_kwargs, retrieve_top_k, result_filename

    if args.device.startswith("cuda"):
        # Check if GPU is available
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please check your CUDA installation.")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
    else:
        assert args.device == "cpu", "Invalid device. Choose from 'cuda' or 'cpu'."
        device = torch.device("cpu")

    torch.cuda.set_device(device)

    # Dataset
    train_dataset = lm_datasets["train"]

    train_batch_size, test_batch_size = 6, 6

    train_dataset = train_dataset.shuffle(seed=args.seed).select(range(int(1_000_000_000 / block_size)))
    if args.debug: # toy dataset
        train_dataset = train_dataset.select(range(int(1_000_000 / block_size)))


    train_sampler = SubsetSampler(range(len(train_dataset)))
    train_dataloader = DataLoader(
        train_dataset, collate_fn=default_data_collator, batch_size=train_batch_size, sampler=train_sampler
    )

    throughput_stats = {}

    sparsifier_kwargs, projector_kwargs = setup_compression_kwargs(args, device)

    # Logging setting
    logger.info(f"The train dataset length: {len(train_dataset)}.")
    logger.info(f"The train batch size: {train_batch_size}")
    logger.info(f"TDA Method: {args.baseline}-{args.tda}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Sparsifier: {sparsifier_kwargs}")
    logger.info(f"Projector: {projector_kwargs}")
    logger.info(f"Layer: {args.layer}")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info("***** Running attribution *****")

    score, profile = None, None
    if args.baseline == "GC":
        check_min_version("4.46.0")
        from _GradComp.utils.common import find_layers
        from _GradComp.attributor.attributor import IFAttributor

        # get which Hessian to use
        tda, hessian = args.tda.split("-")
        hessian = hessian.lower()
        assert tda == "IF", "GradComp only supports Influence Function now."

        layer_names = find_layers(model, args.layer, return_type="name")

        attributor = IFAttributor(
            setting="Llama3_8B_OWT",
            model=model,
            layer_names=layer_names,
            hessian=hessian,
            profile=args.profile,
            device=device,
            sparsifier_kwargs=sparsifier_kwargs,
            projector_kwargs=projector_kwargs,
            offload="disk",
            cache_dir=args.cache_dir,
            chunk_size=16,
        )

        torch.cuda.synchronize(device)
        start_time = time.time()

        if args.mode == "cache":
            result = attributor.cache_gradients(
                    train_dataloader,
                    worker=args.worker,
                )
            if args.profile:
                profile = result[1]

        elif args.mode == "precondition":
            attributor.compute_preconditioners()

        elif args.mode == "ifvp":
            result = attributor.compute_ifvp(worker=args.worker)
            if args.profile:
                profile = result[1]

        elif args.mode == "self_influence":
            score = attributor.compute_self_attribution()

        elif args.mode == "attribute":
            prompt_dataset = FilePromptDataset("./prompts/", tokenizer, block_size)
            test_dataloader = DataLoader(
                prompt_dataset,
                collate_fn=lambda batch: prompt_collate_fn(batch, tokenizer),
                batch_size=test_batch_size,
                shuffle=False
            )
            # Start recording memory snapshot history
            start_record_memory_history()
            result = attributor.attribute(test_dataloader=test_dataloader)

            # Create the memory snapshot file
            export_memory_snapshot()

            # Stop recording memory snapshot history
            stop_record_memory_history()
            exit()

            if args.profile:
                score, profile = result
            else:
                score = result

        elif args.mode == "quant":
            logger.info("Generating the response for each prompt...")
            response_output_dir = os.path.join(f"./results/{args.baseline}/{args.tda}/{args.layer}/response/")
            generate_responses(
                model,
                tokenizer,
                prompt_dataset,
                response_output_dir,
                device=device,
                max_new_tokens=500
            )

            logger.info(f"Retrieving the top 100 influential examples for each prompt...")
            topk_output_dir = os.path.join(f"./results/{args.baseline}/{args.tda}/{args.layer}/topk/")
            retrieve_top_k(
                score,
                k=5,
                prompt_dataset=prompt_dataset,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                output_dir=topk_output_dir
            )
        else:
            raise ValueError("Invalid mode. Choose from 'cache', 'precondition', 'ifvp', 'self_influence', 'attribute', and 'quant'.")

        torch.cuda.synchronize(device)
        end_time = time.time()

    elif args.baseline == "LogIX": #Only used for comparing the throughput, so some of the code are sloppy (specifically, how we get the subset of dataloader)
        from _LogIX.huggingface import LogIXArguments, patch_trainer

        # get which Hessian to use
        tda, hessian = args.tda.split("-")
        hessian = hessian.lower()
        assert tda == "IF", "LogIX only supports Influence Function now."
        assert hessian in ["none", "raw", "kfac", "ekfac"], "Invalid Hessian type."
        assert args.layer == "Linear", "LogIX only supports Linear setting now."
        assert args.projection is not None, "LogIX requires projection method."

        model_cpy = model
        if args.cache:
            LogIXTrainer = patch_trainer(transformers.Trainer)

            model = model.to(device)
            model.eval()

            logix_args_train = LogIXArguments(
                project=f"./LogIX/{args.projection}",
                config=f"./LogIX/{args.projection}.yaml",
                lora=True,
                hessian=hessian,
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

        model = model_cpy # Reset the model to the original one (removing LoRA)
        if args.precondition:
            raise NotImplementedError("Standalone Precondition is not implemented for LogIX.")

        model = model_cpy # Reset the model to the original one (removing LoRA)
        if args.attribute:
            model = model.to(device)
            model.eval()
            logix_args_test = LogIXArguments(
                project=f"./LogIX/{args.projection}",
                config=f"./LogIX/{args.projection}.yaml",
                lora=True,
                hessian=hessian,
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
                train_dataset=prompt_dataset,
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
        raise ValueError("Invalid baseline implementation method. Choose from 'GC' and 'LogIX'.")

    # Calculate throughput
    duration = end_time - start_time

    if args.mode == "cache":
        train_tokens = block_size * len(train_dataset) / int(args.worker.split("/")[1])
        throughput_stats["cache"] = {
            "tokens": train_tokens,
            "duration_seconds": duration,
            "throughput_tokens_per_second": train_tokens / duration
        }
    elif args.mode == "precondition":
        throughput_stats["precondition"] = {
            "duration_seconds": duration,
            "throughput_layers_per_second": len(layer_names) / duration
        }
    elif args.mode == "ifvp":
        throughput_stats["ifvp"] = {
            "train_pairs": len(train_dataset),
            "duration_seconds": duration,
            "throughput_datapoint_per_second": len(train_dataset) / duration
        }
    elif args.mode == "self_influence":
        throughput_stats["self_influence"] = {
            "train_pairs": len(train_dataset),
            "duration_seconds": duration,
            "throughput_pair_per_second": len(train_dataset) / duration
        }
    elif args.mode == "attribute":
        train_test_pairs = len(train_dataset) * len(prompt_dataset)
        throughput_stats["self_influence"] = {
            "train_test_pairs": train_test_pairs,
            "duration_seconds": duration,
            "throughput_pair_per_second": train_test_pairs / duration
        }

    logger.info("***** Attribution finished *****")

    result = {"score": score, "profile": profile, "throughput": throughput_stats} #TODO: Change file name for different mode (otherwise they'll collapse)
    logger.info(result)

    if not args.debug: # only save the results when not in debug mode
        torch.save(result, result_filename(args))

if __name__ == "__main__":
    main()
