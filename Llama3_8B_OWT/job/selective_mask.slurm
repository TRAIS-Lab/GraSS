#!/bin/bash

#SBATCH --job-name=Llama3-SelMask
#SBATCH --output=../log/SelectiveMask_%j.log

echo "Job is starting on `hostname`"

cd ~/Project/GraSS/Llama3_8B_OWT

# Define the dataset and model parameters
DATASET_NAME="openwebtext"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
BLOCK_SIZE=1024
LAYER="Linear"
DEVICE="cuda:0"

for SPAR_DIM in "16" "32" "64" "128" ; do
    python SelectiveMask.py \
        --dataset_name $DATASET_NAME \
        --trust_remote_code \
        --model_name_or_path $MODEL_NAME \
        --output_dir "./checkpoints" \
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

echo "All Selective Mask tasks completed"
