from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator, List

import torch
from torch.utils.data import Sampler, Dataset

import os
import json
import heapq

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

class FilePromptDataset(Dataset):
    def __init__(self, prompt_dir, tokenizer, block_size):
        self.tokenized_prompts = []
        self.raw_prompts = []
        self.file_indices = []

        # Read all prompt files from the directory
        for filename in sorted(os.listdir(prompt_dir)):
            if filename.isdigit() or (filename.endswith('.txt') and filename[:-4].isdigit()):
                file_index = int(filename.split('.')[0])
                file_path = os.path.join(prompt_dir, filename)

                with open(file_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()

                # Store the raw prompt and its file index
                self.raw_prompts.append(prompt)
                self.file_indices.append(file_index)

                # Tokenize the prompt
                inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True,
                                  max_length=block_size)

                # Create a dictionary with input_ids and attention_mask
                self.tokenized_prompts.append({
                    "input_ids": inputs["input_ids"][0],
                    "attention_mask": inputs["attention_mask"][0] if "attention_mask" in inputs else None
                })

    def __len__(self):
        return len(self.tokenized_prompts)

    def __getitem__(self, idx):
        return self.tokenized_prompts[idx]

    def get_raw_prompt(self, idx):
        """Returns the raw text of the prompt at the given index."""
        return self.raw_prompts[idx]

    def get_file_index(self, idx):
        """Returns the file index of the prompt at the given index."""
        return self.file_indices[idx]

def setup_compression_kwargs(args, device):
    if args.sparsification is None:
        sparsifier_kwargs = None
    else:
        sparsification_method, sparsification_dim = args.sparsification.split("-")
        if "*" in sparsification_dim:
            sparsification_factorize = True
            sparsification_dim = sparsification_dim.split("*")
            assert sparsification_dim[0] == sparsification_dim[1], "Sparsification dimension must be the same for factorized projection."

            sparsification_dim = int(sparsification_dim[0])
        else:
            sparsification_factorize = False
            sparsification_dim = int(sparsification_dim)

        sparsifier_kwargs = {
            "proj_dim": sparsification_dim,
            "proj_max_batch_size": 64,
            "proj_seed": args.seed,
            "proj_factorize": sparsification_factorize,
            "device": device,
            "method": sparsification_method,
            "use_half_precision": False,
        }

    if args.projection is None:
        projector_kwargs = {
            "proj_dim": -1,
            "proj_max_batch_size": -1,
            "proj_seed": args.seed,
            "proj_factorize": False,
            "device": device,
            "method": "Identity",
            "use_half_precision": False,
        }
    else:
        proj_method, proj_dim = args.projection.split("-")
        if "*" in proj_dim:
            proj_factorize = True
            proj_dim = proj_dim.split("*")
            assert proj_dim[0] == proj_dim[1], "Projection dimension must be the same for factorized projection."

            proj_dim = int(proj_dim[0])
        else:
            proj_factorize = False
            proj_dim = int(proj_dim)

        projector_kwargs = {
            "proj_dim": proj_dim,
            "proj_max_batch_size": 64,
            "proj_seed": args.seed,
            "proj_factorize": proj_factorize,
            "device": device,
            "method": proj_method,
            "use_half_precision": False,
        }

    # Compatibility checking
    if proj_method == "Localize":
        assert args.baseline == "GC", "Localize option only works with GC baseline."
        assert args.layer == "Linear", "Localize option only works with Linear layer."

    return sparsifier_kwargs, projector_kwargs

def prompt_collate_fn(batch, tokenizer):
    """
    Custom collate function that handles variable length inputs.

    Args:
        batch: List of items from the dataset
        tokenizer: The tokenizer object for padding
    """
    max_length = max(item["input_ids"].size(0) for item in batch)

    # If no pad_token_id is set, use a default value (usually 0)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # Check if eos_token_id is available and use it
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            pad_token_id = tokenizer.eos_token_id
        else:
            # Default to 0 if no other tokens are available
            pad_token_id = 0
        print(f"Warning: pad_token_id is None. Using {pad_token_id} as a replacement.")

    input_ids = []
    attention_mask = []

    for item in batch:
        padding_length = max_length - item["input_ids"].size(0)

        # Pad input_ids
        padded_input_ids = torch.cat([
            item["input_ids"],
            torch.ones(padding_length, dtype=torch.long) * pad_token_id
        ])
        input_ids.append(padded_input_ids)

        # Pad attention_mask if it exists
        if item["attention_mask"] is not None:
            padded_attention_mask = torch.cat([
                item["attention_mask"],
                torch.zeros(padding_length, dtype=torch.long)
            ])
            attention_mask.append(padded_attention_mask)

    batch_dict = {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(input_ids).clone()  # Use the same input_ids as labels for attribution
    }

    if attention_mask:
        batch_dict["attention_mask"] = torch.stack(attention_mask)

    return batch_dict

def generate_responses(model, tokenizer, prompt_dataset, output_dir, device="cuda", max_new_tokens=200, temperature=0.7):
    """
    Generate text responses for each prompt in the dataset and save to files.

    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer associated with the model
        prompt_dataset: Dataset containing prompts
        output_dir: Directory to save responses
        device: Device to run generation on ("cuda" or "cpu")
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling during generation

    Returns:
        List of generated texts
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure model is in evaluation mode and on the correct device
    model.eval()
    model.to(device)

    # Generate text for each prompt and save to files
    generated_texts = []
    for i in range(len(prompt_dataset)):
        prompt = prompt_dataset.get_raw_prompt(i)
        file_idx = prompt_dataset.get_file_index(i)

        # Prepare inputs
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate text
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_texts.append(generated_text)

        # Save response to file
        response_file = os.path.join(output_dir, f"{file_idx}.txt")
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(generated_text)
    return generated_texts

def retrieve_top_k(scores, k=10, prompt_dataset=None, train_dataset=None, tokenizer=None, output_dir=None):
    """
    Find the top k most influential training examples for each test prompt based on attribution scores.
    Optionally save the results to individual files.

    Args:
        scores: Tensor of attribution scores with shape [num_test, num_train]
        k: Number of top examples to return per test prompt
        prompt_dataset: Optional dataset containing the prompts (required if output_dir is specified)
        train_dataset: Optional dataset containing the training examples (required if output_dir is specified and you want to include training text)
        tokenizer: Optional tokenizer to decode training examples (required if train_dataset is provided)
        output_dir: Optional directory to save the results to individual files

    Returns:
        List of lists, where each inner list contains tuples (train_idx, score) for the top k
        influential examples for each test prompt
    """
    if not isinstance(scores, torch.Tensor):
        raise ValueError("Scores must be a tensor with shape [num_test, num_train]")

    # Ensure scores is shaped correctly [num_test, num_train]
    if len(scores.shape) != 2:
        raise ValueError(f"Scores tensor must be 2D, got shape {scores.shape}")

    # Convert to correct orientation if needed (should be [num_test, num_train])
    if scores.shape[0] > scores.shape[1]:  # If it's in the form [num_train, num_test]
        scores = scores.T

    num_test, num_train = scores.shape

    # For each test prompt, find the top k most influential training examples
    top_k_per_prompt = []

    # Create output directory if specified
    if output_dir is not None:
        if prompt_dataset is None:
            raise ValueError("prompt_dataset must be provided if output_dir is specified")
        os.makedirs(output_dir, exist_ok=True)

    # Check if we have training text capability
    include_training_text = (train_dataset is not None and tokenizer is not None)

    for test_idx in range(num_test):
        # Get scores for this test prompt
        test_scores = scores[test_idx].abs().cpu().numpy()  # Use absolute value for influence

        # Get indices of top k training examples
        top_indices = heapq.nlargest(min(k, num_train), range(num_train), key=lambda i: test_scores[i])

        # Create list of (train_idx, score) tuples
        prompt_top_k = [(train_idx, float(test_scores[train_idx])) for train_idx in top_indices]

        top_k_per_prompt.append(prompt_top_k)

        # Save to file if output_dir is specified
        if output_dir is not None:
            # Get the file index corresponding to this prompt
            file_idx = prompt_dataset.get_file_index(test_idx)

            # Create a JSON file for this prompt
            influential_file = os.path.join(output_dir, f"{file_idx}.json")

            # Create the influential examples with optional training text
            influential_examples = []
            for train_idx, score in prompt_top_k:
                example_dict = {"train_idx": train_idx, "score": float(score)}

                # Include training text if possible
                if include_training_text:
                    try:
                        example = train_dataset[train_idx]
                        training_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
                        example_dict["training_text"] = training_text
                    except Exception as e:
                        print(f"Error decoding training example {train_idx}: {e}")

                influential_examples.append(example_dict)

            with open(influential_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "prompt_idx": test_idx,
                    "prompt_text": prompt_dataset.get_raw_prompt(test_idx),
                    "file_index": file_idx,
                    "influential_examples": influential_examples
                }, f, indent=2)

    return top_k_per_prompt

def result_filename(args):
    if args.sparsification is not None:
        sparsification_name = args.sparsification
    else:
        sparsification_name = "NA"

    if args.projection is not None:
        projection_name = args.projection
    else:
        projection_name = "NA"

    # also add worker's name
    worker_id, total_worker = args.worker.split('/')
    file_name = f"./results/{args.baseline}/{args.tda}/{args.layer}/{sparsification_name}->{projection_name}_{args.mode}_worker{worker_id}_of_{total_worker}.pt"

    return file_name