#!/bin/bash

#SBATCH --job-name=MNIST-SelectiveMask
#SBATCH --output=../log/SelectiveMask_%j.log

echo "Job is starting on `hostname`"

cd ~/Project/GraSS/MLP_MNIST

for SPAR_DIM in "2048" "4096" "8192" ; do
	python SelectiveMask.py \
		--device cuda:0 \
		--sparsification_dim $SPAR_DIM \
		--epoch 5000 \
		--n 5000 \
		--log_interval 500 \
		--learning_rate 5e-5 \
		--regularization 1e-6 \
		--early_stop 0.9 \
		--output_dir "./SelectiveMask" \
		--seed 62
done

echo "All tasks completed"