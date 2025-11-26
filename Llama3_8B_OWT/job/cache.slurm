#!/bin/bash
# Cache mode automatically runs preconditioners and IFVP computation
# when hessian="eFIM", so no separate precondition/ifvp scripts are needed.

#SBATCH --job-name=Llama3-Cache
#SBATCH --output=../log/cache_%A_%a.log
#SBATCH --array=0-19

echo "Job is starting on `hostname`"

cd ~/Project/GraSS/Llama3_8B_OWT

# Define the dataset and model parameters
DATASET_NAME="openwebtext"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
BLOCK_SIZE=1024
DEVICE="cuda:0"
CACHE_DIR="./cache/"

# Worker configuration from SLURM array
WORKER_ID=$SLURM_ARRAY_TASK_ID
TOTAL_WORKERS=20

# Compression settings
SPARSIFIER="random_mask"
SPAR_DIM="128*128"
PROJECTOR="sjlt"
PROJ_DIM="4096"

python attribute.py \
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
    --mode "cache" \
    --cache_dir $CACHE_DIR \
    --worker "$WORKER_ID/$TOTAL_WORKERS" \
    --profile

echo "Cache task $WORKER_ID/$TOTAL_WORKERS completed"
