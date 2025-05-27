# GraSS 🌿

This is the official implementation of [GraSS: Scalable Influence Function with Sparse Gradient Compression](https://arxiv.org/abs/2505.18976v1).

## Setup Guide

Please follow the installation guide from [dattri](https://github.com/TRAIS-Lab/dattri) in order to correctly install `fast_jl`. By installing dattri, all the basic libraries will also be installed, and you should be able to run all experiments *except for LM-related* ones such as GPT2 and Llama3-8B. For those, you'll need to have the usual Hugging Face libraries such as `datasets` installed as well.

## File Structure

The folders either correspond to *libraries* or *experiments*; specifically, the ones starting with `_` are *libraries* (or baselines) that implement the data attribution algorithms, while others correspond to *experiments*. In particular, there are four libraries:

1. `_GradComp`: The main implementation supports influence function with linear layer's gradient factorized compression. In particular, **FactGraSS** (and **SJLT** in `_GradComp/projection/sjlt`).
2. `_dattri`: The [dattri](https://github.com/TRAIS-Lab/dattri) library with **GraSS** implementations in `_dattri/func/projection.py`.
3. `_SelectiveMask`: The implementation of **Selective Mask**.
4. `_LogIX`: The [LogIX](https://github.com/logix-project/logix) library with some efficiency fixes to cross-validate our LoGra implementation.

## Quick Start

We provide the scripts for the experiments.

>Note that in the codebase, we call *Random Mask* as *Random*.

### MLP+MNIST/ResNet+CIFAR/MusicTransformer+MAESTRO

In these settings, the LDS results and the models are provided by dattri, so we don't need to train models ourselves. To obtain all the results for, e.g., MLP+MNIST, run the following scripts:

1. Selective Mask:
	```bash
	for PROJ_DIM in "2048" "4096" "8192" ; do
		python SelectiveMask.py \
			--device "cuda" \
			--sparsification_dim $PROJ_DIM \
			--epoch 5000 \
			--n 5000 \
			--log_interval 500 \
			--learning_rate 5e-5 \
			--regularization 1e-6 \
			--early_stop 0.9 \
			--output_dir "./SelectiveMask"
	done
	```
2. Attribution:
	```bash
	for PROJ_DIM in "2048" "4096" "8192" ; do
		for PROJ_METHOD in "Random" "SelectiveMask" "SJLT" "FJLT" "Gaussian"; do
			python score.py \
				--device "cuda" \
				--proj_method $PROJ_METHOD \
				--proj_dim $PROJ_DIM \
				--seed 22
		done
	done
	```

### GPT2+Wikitext

For GPT2 experiments, since the LDS result and the fine-tuned models are not available, we need to manually produce them first. Consider running the following scripts:

1. Fine-tune 50 models:
	```bash
	# Loop over the task IDs
	for SLURM_ARRAY_TASK_ID in {0..49}; do
		echo "Starting task ID: $SLURM_ARRAY_TASK_ID"

		# Set the output directory and seed based on the current task ID
		OUTPUT_DIR="./checkpoints/${SLURM_ARRAY_TASK_ID}"
		SEED=${SLURM_ARRAY_TASK_ID}

		# Create the output directory
		mkdir -p $OUTPUT_DIR

		# Run the training script
		python train.py \
			--dataset_name "wikitext" \
			--dataset_config_name "wikitext-2-raw-v1" \
			--model_name_or_path "openai-community/gpt2" \
			--output_dir $OUTPUT_DIR \
			--block_size 512 \
			--subset_ratio 0.5 \
			--seed $SEED

		echo "Task ID $SLURM_ARRAY_TASK_ID completed"
	done
	```
2. Obtain groundtruth for computing LDS:
	```bash
	python groundtruth.py\
		--dataset_name "wikitext" \
		--dataset_config_name "wikitext-2-raw-v1" \
		--model_name_or_path "openai-community/gpt2" \
		--output_dir ./checkpoints \
		--block_size 512 \
		--seed 0
	```
3. Selective Mask training:
	```bash
	for PROJ_DIM in "32" "64" "128" ; do
		python SelectiveMask.py\
			--dataset_name "wikitext" \
			--dataset_config_name "wikitext-2-raw-v1" \
			--model_name_or_path "openai-community/gpt2" \
			--output_dir "./checkpoints" \
			--block_size 512 \
			--seed 0 \
			--device "cuda" \
			--layer "Linear" \
			--sparsification_dim $PROJ_DIM \
			--epoch 500 \
			--learning_rate 1e-5 \
			--regularization 5e-5 \
			--early_stop 0.9 \
			--log_interval 100 \
			--n 200
	done
	```
4. Attribution: The following is an example for FactGraSS. To test other compression method, e.g., LoGra, simply remove `--sparsification Random-128*128` and change `--projection SJLt-4096` to `--projection Gaussian-64*64`.
	```bash
	python score.py\
		--dataset_name "wikitext" \
		--dataset_config_name "wikitext-2-raw-v1" \
		--model_name_or_path "openai-community/gpt2" \
		--output_dir "./checkpoints" \
		--block_size 512 \
		--seed 0 \
		--device "cuda" \
		--baseline "GC" \
		--tda "IF-RAW" \
		--layer "Linear" \
		--sparsification Random-128*128 \
		--projection SJLT-4096 \
		--val_ratio 0.1 \
		--profile
	```

### Llama3-8B+OpenWebText

For billion-scale model, since we do not need to do quantitative experiment, we do not need to fine-tune the model several times. Here, since this is a large scale experiment, we divide the attribution into several phases, specified by `--mode`. In order, the available options are `cache`, `precondition`, `ifvp`, `attribute`. Furthermore, for `cache` and `ifvp`, we provide a further `--worker` argument to parallelize the job by splitting the dataset among several job instances.

> We note that the complete order is `cache`→`precondition`→`ifvp`→`attribute`

Here, we provide an example for `cache` and `attribute`:

1. `cache`/`ifvp`: For caching projected gradients and computing iFVP, we can parallelize via `--worker`. The following script submits 20 jobs to divide the dataset into 20 chunks:

	```bash
	#SBATCH -a 0-19

	WORKER_ID=$SLURM_ARRAY_TASK_ID

	python attribute.py \
		--dataset_name "openwebtext" \
		--trust_remote_code \
		--model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
		--output_dir "./checkpoints" \
		--block_size 1024 \
		--seed 0 \
		--device "cuda" \
		--baseline "GC" \
		--tda "IF-RAW" \
		--layer "Linear" \
		--sparsification "Random-128*128" \
		--projection "SJLT-4096" \
		--mode "cache" \ # or ifvp
		--cache_dir "./cache/" \
		--worker "$WORKER_ID/20" \
		--profile
	```
2. `attribute`/`precondition`: Computing preconditioners and also attributing do  not have the parallelization functionality. Simply removing `--worker` and change `--mode`:

	```bash
	python attribute.py \
		--dataset_name "openwebtext" \
		--trust_remote_code \
		--model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
		--output_dir "./checkpoints" \
		--block_size 1024 \
		--seed 0 \
		--device "cuda" \
		--baseline "GC" \
		--tda "IF-RAW" \
		--layer "Linear" \
		--sparsification "Random-128*128" \
		--projection "SJLT-4096" \
		--mode "attribute" \ # or precondition
		--cache_dir "./cache/" \
		--profile
	```

## Citation

If you find this repository valuable, please give it a star! Got any questions or feedback? Feel free to open an issue. Using this in your work? Please reference us using the provided citation:
```bibtex
@misc{hu2025grass,
  author        = {Pingbang Hu and Joseph Melkonian and Weijing Tang and Han Zhao and Jiaqi W. Ma},
  title         = {GraSS: Scalable Influence Function with Sparse Gradient Compression},
  archiveprefix = {arXiv},
  eprint        = {2505.18976},
  primaryclass  = {cs.LG},
  url           = {https://arxiv.org/abs/2505.18976},
  year          = {2025}
}
```
