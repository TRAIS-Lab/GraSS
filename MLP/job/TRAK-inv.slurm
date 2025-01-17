#!/bin/bash
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=TRAK-O

echo "job is starting on `hostname`"

cd ~/Project/Sparse-Influence

activation_functions=("relu" "sigmoid" "tanh" "leaky_relu" "linear")
high_dimension=("512" "1024" "2048" "4096" "8192")
low_dimension=("64" "128" "256")
eps_list=("0.0000001" "0.000001" "0.00001" "0.0001" "0.001")
drop_rate_list=("0.1" "0.2" "0.3" "0.4" "0.5")

# for activation_fn in "${activation_functions[@]}"; do
# 	# Train Model
#     python3 retrain.py --dataset mnist --model mlp --mode lds --seed 1 --extra_param activation_fn="$activation_fn"

# 	# Calculate Groundtruth LDS
# 	python3 groundtruth.py --model mlp --method TRAK-50-inv --extra_param activation_fn="$activation_fn"
# done

# Final Benchmark (TRAK-50-inv)
for activation_fn in "${activation_functions[@]}"; do
	# Original
	for proj_dim in "${low_dimension[@]}"; do
		python3 benchmark.py --dataset mnist --model mlp --method TRAK-50-inv --metric lds --device cpu --sparse_check --extra_param activation_fn="$activation_fn" --extra_param proj_dim="$proj_dim"
	done
	for proj_dim in "${high_dimension[@]}"; do
		python3 benchmark.py --dataset mnist --model mlp --method TRAK-50-inv --metric lds --sparse_check --extra_param activation_fn="$activation_fn" --extra_param proj_dim="$proj_dim"
	done

	# Sparsify: Threshold
	# for eps in "${eps_list[@]}"; do
	# 	for proj_dim in "${low_dimension[@]}"; do
	# 		python3 benchmark.py --dataset mnist --model mlp --method TRAK-50-inv --metric lds --device cpu --sparse_check --sparsify threshold-"$eps" --extra_param activation_fn="$activation_fn" --extra_param proj_dim="$proj_dim"
	# 	done
	# 	for proj_dim in "${high_dimension[@]}"; do
	# 		python3 benchmark.py --dataset mnist --model mlp --method TRAK-50-inv --metric lds --sparse_check --sparsify threshold-"$eps" --extra_param activation_fn="$activation_fn" --extra_param proj_dim="$proj_dim"
	# 	done
	# done

	# Sparsify: Random Drop
	# for drop_rate in "${drop_rate_list[@]}"; do
	# 	for proj_dim in "${low_dimension[@]}"; do
	# 		python3 benchmark.py --dataset mnist --model mlp --method TRAK-50-inv --metric lds --device cpu --sparse_check --sparsify random-"$drop_rate" --extra_param activation_fn="$activation_fn" --extra_param proj_dim="$proj_dim"
	# 	done
	# 	for proj_dim in "${high_dimension[@]}"; do
	# 		python3 benchmark.py --dataset mnist --model mlp --method TRAK-50-inv --metric lds --sparse_check --sparsify random-"$drop_rate" --extra_param activation_fn="$activation_fn" --extra_param proj_dim="$proj_dim"
	# 	done
	# done
done