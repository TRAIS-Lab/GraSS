#!/bin/bash

#SBATCH --job-name=GPT2-Train
#SBATCH --output=../log/train_%j.log

echo "Job is starting on `hostname`"

cd ~/Project/GraSS/GPT2_wikitext

# Define the dataset and model parameters
DATASET_NAME="wikitext"
DATASET_CONFIG_NAME="wikitext-2-raw-v1"
MODEL_NAME="openai-community/gpt2"
BLOCK_SIZE=512
SUBSET_RATIO=0.5

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
        --dataset_name $DATASET_NAME \
        --dataset_config_name $DATASET_CONFIG_NAME \
        --model_name_or_path $MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --block_size $BLOCK_SIZE \
        --subset_ratio $SUBSET_RATIO \
        --seed $SEED

    echo "Task ID $SLURM_ARRAY_TASK_ID completed"
done

echo "All tasks completed"