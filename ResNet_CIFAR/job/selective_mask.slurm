#!/bin/bash

#SBATCH --job-name=CIFAR-SelectiveMask
#SBATCH --output=../log/SelectiveMask_%j.log

echo "Job is starting on `hostname`"

cd ~/Project/GraSS/ResNet_CIFAR

for SPAR_DIM in "2048" "4096" "8192" ; do
	python SelectiveMask.py \
		--device cuda:3 \
		--sparsification_dim $SPAR_DIM \
		--epoch 4000 \
		--n 4000 \
		--log_interval 500 \
		--learning_rate 5e-5 \
		--regularization 1e-7 \
		--early_stop 0.9 \
		--output_dir "./SelectiveMask" \
		--seed 2
done

echo "All tasks completed"