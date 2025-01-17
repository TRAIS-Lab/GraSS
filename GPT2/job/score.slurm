#!/bin/bash

#SBATCH --job-name=GPT2
#SBATCH --output=../logs/score_%j.log

echo "Job is starting on `hostname`"

cd ~/Project/Sparse-Influence/GPT2

# Define the dataset and model parameters
DATASET_NAME="wikitext"
DATASET_CONFIG_NAME="wikitext-2-raw-v1"
MODEL_NAME="openai-community/gpt2"
BLOCK_SIZE=512
# PROJ="FJLT"

python score.py\
    --dataset_name $DATASET_NAME \
    --dataset_config_name $DATASET_CONFIG_NAME \
    --model_name_or_path $MODEL_NAME \
    --output_dir ./checkpoints \
    --block_size $BLOCK_SIZE \
    --seed 0

python score.py\
    --dataset_name $DATASET_NAME \
    --dataset_config_name $DATASET_CONFIG_NAME \
    --model_name_or_path $MODEL_NAME \
    --output_dir ./checkpoints \
    --block_size $BLOCK_SIZE \
	--proj "FJLT"\
    --seed 0

echo "All tasks completed"