#!/bin/bash

#SBATCH --job-name=MAESTRO-score
#SBATCH --output=../log/score_%j.log

echo "Job is starting on `hostname`"

cd ~/Project/GraSS/MusicTransformer_MAESTRO

for PROJ_DIM in "2048"  ; do
	for PROJ_TYPE in "random_mask" "selective_mask" "sjlt" "grass_4" "selective_grass_4" "fjlt"; do
		python score.py \
			--device "cuda" \
			--proj_type $PROJ_TYPE \
			--proj_dim $PROJ_DIM \
			--seed 42
	done
done

echo "All tasks completed"