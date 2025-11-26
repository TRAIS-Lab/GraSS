#!/bin/bash

#SBATCH --job-name=GPT2-GT
#SBATCH --output=../log/groundtruth_%j.log

echo "Job is starting on `hostname`"

cd ~/Project/GraSS/GPT2_wikitext

# Define the dataset and model parameters
DATASET_NAME="wikitext"
DATASET_CONFIG_NAME="wikitext-2-raw-v1"
MODEL_NAME="openai-community/gpt2"
BLOCK_SIZE=512

python groundtruth.py\
    --dataset_name $DATASET_NAME \
    --dataset_config_name $DATASET_CONFIG_NAME \
    --model_name_or_path $MODEL_NAME \
    --output_dir ./checkpoints \
    --block_size $BLOCK_SIZE \
    --seed 0

echo "All tasks completed"