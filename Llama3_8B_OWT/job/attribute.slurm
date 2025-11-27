#!/bin/bash

#SBATCH --job-name=Llama3-Attr
#SBATCH --output=../log/attribute_%j.log

echo "Job is starting on `hostname`"

cd ~/Project/GraSS/Llama3_8B_OWT

# Define the dataset and model parameters
DATASET_NAME="openwebtext"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
BLOCK_SIZE=1024
DEVICE="cuda:0"
CACHE_DIR="./cache/"

# Compression settings
SPARSIFIER="random_mask"
SPAR_DIM="128*128"
PROJECTOR="sjlt"
PROJ_DIM="4096"

python score.py \
    --dataset_name $DATASET_NAME \
    --trust_remote_code \
    --model_name_or_path $MODEL_NAME \
    --output_dir "./checkpoints" \
    --block_size $BLOCK_SIZE \
    --seed 0 \
    --device $DEVICE \
    --baseline "dattri" \
    --hessian "eFIM" \
    --layer "Linear" \
    --sparsification $SPARSIFIER-$SPAR_DIM \
    --projection $PROJECTOR-$PROJ_DIM \
    --mode "attribute" \
    --cache_dir $CACHE_DIR \
    --profile

echo "Attribution task completed"
