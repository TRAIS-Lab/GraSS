#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void sjlt_projection_kernel(
    const scalar_t* input,           // Input tensor [batch_size, original_dim]
    scalar_t* output,                // Output tensor [batch_size, proj_dim]
    const int64_t* rand_indices,     // Random indices [original_dim, c]
    const int8_t* rand_signs,        // Random signs [original_dim, c]
    const int batch_size,
    const int original_dim,
    const int proj_dim,
    const int c) {

    // Each block now processes multiple chunks of the input
    // Calculate dimensions per block and assign work accordingly
    const int total_threads = gridDim.x * blockDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate how many dimensions each thread needs to process
    const int dims_per_thread = (original_dim + total_threads - 1) / total_threads;

    // Process multiple dimensions per thread
    for (int chunk = 0; chunk < dims_per_thread; chunk++) {
        // Calculate the actual dimension index for this thread and chunk
        const int idx = thread_id + chunk * total_threads;

        // Skip if beyond the original dimension size
        if (idx >= original_dim) continue;

        // Load the random indices and signs for this dimension (idx)
        int local_rand_indices[16];  // Assuming c <= 16, adjust if needed
        int8_t local_rand_signs[16]; // Assuming c <= 16, adjust if needed

        for (int j = 0; j < c; j++) {
            local_rand_indices[j] = rand_indices[idx * c + j];
            local_rand_signs[j] = rand_signs[idx * c + j];
        }

        // Process each sample in the batch
        for (int b = 0; b < batch_size; b++) {
            scalar_t val = input[b * original_dim + idx];

            // Only process if value is non-zero
            if (val != 0) {
                for (int j = 0; j < c; j++) {
                    int output_idx = b * proj_dim + local_rand_indices[j];
                    scalar_t scaled_val = val * local_rand_signs[j];

                    // Atomic add to handle race conditions when multiple threads update the same output location
                    atomicAdd(&output[output_idx], scaled_val);
                }
            }
        }
    }
}

// Normalize kernel
template <typename scalar_t>
__global__ void normalize_kernel(
    scalar_t* output,     // Output tensor [batch_size, proj_dim]
    const int batch_size,
    const int proj_dim,
    const float normalization_factor) {

    const int total_threads = gridDim.x * blockDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * proj_dim;

    // Calculate elements per thread
    const int elements_per_thread = (total_elements + total_threads - 1) / total_threads;

    // Process multiple elements per thread
    for (int chunk = 0; chunk < elements_per_thread; chunk++) {
        const int idx = thread_id + chunk * total_threads;
        if (idx < total_elements) {
            output[idx] = output[idx] * normalization_factor;
        }
    }
}

// Function to set the cache configuration for our kernels
void setCacheConfig() {
    // Set L1 cache preference for the projection kernel
    cudaFuncSetCacheConfig(sjlt_projection_kernel<float>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(sjlt_projection_kernel<double>, cudaFuncCachePreferL1);

    // Set L1 cache preference for the normalize kernel
    cudaFuncSetCacheConfig(normalize_kernel<float>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(normalize_kernel<double>, cudaFuncCachePreferL1);
}

// C++ wrapper for the CUDA kernel with fixed block count
std::vector<torch::Tensor> sjlt_projection_cuda(
    torch::Tensor input,
    torch::Tensor rand_indices,
    torch::Tensor rand_signs,
    int proj_dim,
    int c,
    int threads,
    int fixed_blocks) {

    // Set the cache configuration (call this once)
    static bool cacheConfigSet = false;
    if (!cacheConfigSet) {
        setCacheConfig();
        cacheConfigSet = true;
    }

    auto batch_size = input.size(0);
    auto original_dim = input.size(1);

    // Create output tensor
    auto output = torch::zeros({batch_size, proj_dim},
                              torch::TensorOptions()
                              .dtype(input.dtype())
                              .device(input.device()));

    // Compute normalization factor
    float normalization_factor = 1.0f / sqrt(c);

    // Fix the number of blocks to the specified value
    // const int fixed_blocks = 168;

    // Ensure threads is a multiple of 32 (warp size) for optimal performance
    threads = (threads / 32) * 32;

    // Launch the kernel with fixed block count
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sjlt_projection_cuda", ([&] {
        sjlt_projection_kernel<scalar_t><<<fixed_blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            rand_indices.data_ptr<int64_t>(),
            rand_signs.data_ptr<int8_t>(),
            batch_size,
            original_dim,
            proj_dim,
            c);

        // Use the same fixed block count for normalization
        normalize_kernel<scalar_t><<<fixed_blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            batch_size,
            proj_dim,
            normalization_factor);
    }));

    return {output};
}

// Define module functions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sjlt_projection_cuda", &sjlt_projection_cuda, "SJLT projection CUDA implementation with fixed block count");
}