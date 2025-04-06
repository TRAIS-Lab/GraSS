#!/bin/bash

#SBATCH --job-name=Figure
#SBATCH --output=score_%j.log

echo "Job is starting on `hostname`"

# cd ~/Project/Sparse-Influence/MLP_MNIST

# for PROJ_DIM in "2048" "4096" "8192" ; do
# 	for PROJ_METHOD in "FJLT" ; do
# 		python score.py \
# 			--device cuda:3 \
# 			--proj_method $PROJ_METHOD \
# 			--proj_dim $PROJ_DIM
# 	done
# done

cd ~/Project/Sparse-Influence/MusicTransformer_MAESTRO

for PROJ_DIM in "2048" "4096" "8192" ; do
	for PROJ_METHOD in "SJLT" "FJLT" ; do
		python score.py \
			--device cuda:0 \
			--proj_method $PROJ_METHOD \
			--proj_dim $PROJ_DIM
	done
done

echo "All tasks completed"