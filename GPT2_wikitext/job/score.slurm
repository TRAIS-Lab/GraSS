#!/bin/bash

#SBATCH --job-name=GPT2-Score
#SBATCH --output=../log/score_%j.log

echo "Job is starting on `hostname`"

cd ~/Project/GraSS/GPT2_wikitext

# Define the dataset and model parameters
DATASET_NAME="wikitext"
DATASET_CONFIG_NAME="wikitext-2-raw-v1"
MODEL_NAME="openai-community/gpt2"
BLOCK_SIZE=512
CHECKPOINTS="./checkpoints/wd=0.0_lr=5e-5"
DEVICE="cuda:0"

# Element of FactGraSS testing
for SPAR_DIM in "16*16" "32*32" "64*64" ; do
	for SPARSIFIER in "random_mask" "selective_mask" "sjlt" ; do
		python score.py\
			--dataset_name $DATASET_NAME \
			--dataset_config_name $DATASET_CONFIG_NAME \
			--model_name_or_path $MODEL_NAME \
			--output_dir $CHECKPOINTS \
			--block_size $BLOCK_SIZE \
			--seed 0 \
			--device $DEVICE \
			--baseline "dattri" \
			--hessian "eFIM" \
			--layer "Linear" \
			--sparsification $SPARSIFIER-$SPAR_DIM \
			--val_ratio 0.1 \
			--profile
	done
done


# FactGraSS
for SPAR_DIM in "128*128" ; do
	for SPARSIFIER in "random_mask" "selective_mask" ; do
		for PROJ_DIM in "256" "1024" "4096" ; do
			for PROJ in "sjlt" ; do
				python score.py\
					--dataset_name $DATASET_NAME \
					--dataset_config_name $DATASET_CONFIG_NAME \
					--model_name_or_path $MODEL_NAME \
					--output_dir $CHECKPOINTS \
					--block_size $BLOCK_SIZE \
					--seed 0 \
					--device $DEVICE \
					--baseline "dattri" \
					--hessian "eFIM" \
					--layer "Linear" \
					--sparsification $SPARSIFIER-$SPAR_DIM \
					--projection $PROJ-$PROJ_DIM \
					--val_ratio 0.1 \
					--profile
			done
		done
	done
done

# LoGra
for SPAR_DIM in "16*16" "32*32" "64*64" ; do
	for SPARSIFIER in "normal" ; do
		python score.py\
			--dataset_name $DATASET_NAME \
			--dataset_config_name $DATASET_CONFIG_NAME \
			--model_name_or_path $MODEL_NAME \
			--output_dir $CHECKPOINTS \
			--block_size $BLOCK_SIZE \
			--seed 0 \
			--device $DEVICE \
			--baseline "dattri" \
			--hessian "eFIM" \
			--layer "Linear" \
			--sparsification $SPARSIFIER-$SPAR_DIM \
			--val_ratio 0.1 \
			--profile
	done
done

# LogIX's implementation of Logra
for SPAR_DIM in "16*16" "32*32" "64*64" ; do
	for SPARSIFIER in "Random" ; do
		python score.py\
			--dataset_name $DATASET_NAME \
			--dataset_config_name $DATASET_CONFIG_NAME \
			--model_name_or_path $MODEL_NAME \
			--output_dir $CHECKPOINTS \
			--block_size $BLOCK_SIZE \
			--seed 0 \
			--device $DEVICE \
			--baseline "LogIX" \
			--hessian "eFIM" \
			--layer "Linear" \
			--sparsification $SPARSIFIER-$SPAR_DIM \
			--val_ratio 0.1 \
			--profile
	done
done


echo "All tasks completed"