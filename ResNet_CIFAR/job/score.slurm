#!/bin/bash

#SBATCH --job-name=CIFAR-score
#SBATCH --output=../log/score_%j.log

echo "Job is starting on `hostname`"

cd ~/Project/GraSS/ResNet_CIFAR

for PROJ_DIM in "2048" "4096" "8192" ; do
	for PROJ_TYPE in "random_mask" "selective_mask" "sjlt" "grass_4" "selective_grass_4" "fjlt"; do
		python score.py \
			--device cuda:0 \
			--proj_type $PROJ_TYPE \
			--proj_dim $PROJ_DIM \
			--seed 42
	done
done

echo "All tasks completed"