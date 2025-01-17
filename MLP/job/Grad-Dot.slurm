#!/bin/bash

#SBATCH --job-name=Grad-Dot-Original

echo "job is starting on `hostname`"

cd ~/Project/Sparse-Influence

activation_functions=("relu" "sigmoid" "tanh" "leaky_relu" "linear")
high_dimension=("512" "1024" "2048" "4096" "8192")
low_dimension=("64" "128" "256")
eps_list=("0.05" "0.01" "0.1" "0.5")
delta_list=("0.05" "0.01" "0.1" "0.2")

# for activation_fn in "${activation_functions[@]}"; do
# 	# Train Model
#     python3 retrain.py --dataset mnist --model mlp --mode lds --seed 1 --subset_ratio 1 --extra_param activation_fn="$activation_fn"

# 	# Calculate Groundtruth LDS
# 	python3 groundtruth.py --model mlp --method Grad-Dot --extra_param activation_fn="$activation_fn"
# done

# Final Benchmark (Grad-Dot)
# for activation_fn in "${activation_functions[@]}"; do
# 	for eps in "${eps_list[@]}"; do
# 		for delta in "${delta_list[@]}"; do
# 			# Original
# 			for proj_dim in "${low_dimension[@]}"; do
# 				python3 benchmark.py --dataset mnist --model mlp --method Grad-Dot --metric lds --device cpu --sparse_check --extra_param activation_fn="$activation_fn" --extra_param proj_dim="$proj_dim" --extra_param eps="$eps" --extra_param delta="$delta"
# 			done
# 			for proj_dim in "${high_dimension[@]}"; do
# 				python3 benchmark.py --dataset mnist --model mlp --method Grad-Dot --metric lds --device cpu --sparse_check --extra_param activation_fn="$activation_fn" --extra_param proj_dim="$proj_dim" --extra_param eps="$eps" --extra_param delta="$delta"
# 			done
# 		done
# 	done
# done


for activation_fn in "${activation_functions[@]}"; do
	# Original
	python3 benchmark.py --dataset mnist --model mlp --method Grad-Dot --metric lds --device cpu --sparse_check --extra_param activation_fn="$activation_fn"
done