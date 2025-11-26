#!/bin/bash

#SBATCH --job-name=MAESTRO-SelectiveMask
#SBATCH --output=../log/SelectiveMask_%j.log

echo "Job is starting on `hostname`"

cd ~/Project/GraSS/MusicTransformer_MAESTRO

for SPAR_DIM in "524288"; do
	python SelectiveMask.py \
		--device cuda:3 \
		--sparsification_dim $SPAR_DIM \
		--epoch 500 \
		--n 3000 \
		--log_interval 100 \
		--learning_rate 1e-5 \
		--regularization 1e-5 \
		--early_stop 0.9 \
		--output_dir "./SelectiveMask" \
		--seed 82
done

echo "All tasks completed"