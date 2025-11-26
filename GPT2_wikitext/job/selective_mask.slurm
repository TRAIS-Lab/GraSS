#!/bin/bash

#SBATCH --job-name=GPT2-SelectiveMask
#SBATCH --output=../log/SelectiveMask_%j.log

echo "Job is starting on `hostname`"

cd ~/Project/GraSS/GPT2_wikitext

# Define the dataset and model parameters
DATASET_NAME="wikitext"
DATASET_CONFIG_NAME="wikitext-2-raw-v1"
MODEL_NAME="openai-community/gpt2"
BLOCK_SIZE=512
CHECKPOINTS="./checkpoints/wd=0.0_lr=5e-5"
LAYER="Linear"
DEVICE="cuda:0"

for SPAR_DIM in "16" "32" "64" "128" ; do
	python SelectiveMask.py\
		--dataset_name $DATASET_NAME \
		--dataset_config_name $DATASET_CONFIG_NAME \
		--model_name_or_path $MODEL_NAME \
		--output_dir $CHECKPOINTS \
		--block_size $BLOCK_SIZE \
		--seed 0 \
		--device $DEVICE \
		--layer $LAYER \
		--sparsification_dim $SPAR_DIM \
		--epoch 500 \
		--learning_rate 1e-5 \
		--regularization 5e-5 \
		--early_stop 0.9 \
		--log_interval 100 \
		--n 200
done

echo "All tasks completed"