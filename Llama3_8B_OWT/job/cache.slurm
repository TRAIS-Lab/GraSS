#!/bin/bash
# Cache mode automatically runs preconditioners and IFVP computation
# when hessian="eFIM", so no separate precondition/ifvp scripts are needed.

#SBATCH --job-name=Llama3-Cache
#SBATCH --output=../log/cache_%A_%a.log
#SBATCH --array=0-19

echo "Job is starting on `hostname`"

cd ~/Project/GraSS/Llama3_8B_OWT

# Define the dataset and model parameters
DATASET_NAME="vietgpt/openwebtext_en" # to avoid legacy python script issue for datasets
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
BLOCK_SIZE="1024"
DEVICE="cuda:0"
CACHE_DIR="/scratch/bfwm/phu1/Project/GraSS/cache/"
DTYPE="float16"

# Worker configuration from SLURM array
WORKER_ID=$SLURM_ARRAY_TASK_ID
TOTAL_WORKERS=1

python score.py \
    --dataset_name $DATASET_NAME \
    --model_name_or_path $MODEL_NAME \
	--dataset_cache_dir "/scratch/bfwm/phu1/Project/GraSS/dataset_cache/" \
    --block_size $BLOCK_SIZE \
    --seed 0 \
    --device $DEVICE \
    --baseline "dattri" \
    --hessian "eFIM" \
    --layer "Linear" \
    --sparsification "random_mask-128*128" \
    --projection "sjlt-4096" \
    --mode "cache" \
    --cache_dir $CACHE_DIR \
    --worker $WORKER_ID/$TOTAL_WORKERS \
	--dtype $DTYPE \
	--throughput_test \
    --profile

python score.py \
    --dataset_name $DATASET_NAME \
    --model_name_or_path $MODEL_NAME \
	--dataset_cache_dir "/scratch/bfwm/phu1/Project/GraSS/dataset_cache/" \
    --block_size $BLOCK_SIZE \
    --seed 0 \
    --device $DEVICE \
    --baseline "dattri" \
    --hessian "eFIM" \
    --layer "Linear" \
    --sparsification "normal-64*64" \
    --mode "cache" \
    --cache_dir $CACHE_DIR \
    --worker $WORKER_ID/$TOTAL_WORKERS \
	--dtype $DTYPE \
	--throughput_test \
    --profile

python score.py \
    --dataset_name $DATASET_NAME \
    --model_name_or_path $MODEL_NAME \
	--dataset_cache_dir "/scratch/bfwm/phu1/Project/GraSS/dataset_cache/" \
    --block_size $BLOCK_SIZE \
    --seed 0 \
    --device $DEVICE \
    --baseline "LogIX" \
    --hessian "eFIM" \
    --layer "Linear" \
    --sparsification "Random-64*64" \
    --mode cache \
    --cache_dir $CACHE_DIR \
    --worker $WORKER_ID/$TOTAL_WORKERS \
	--dtype $DTYPE \
	--throughput_test \
    --profile

echo "Cache task $WORKER_ID/$TOTAL_WORKERS completed"
