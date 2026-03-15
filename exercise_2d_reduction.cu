// ─────────────────────────────────────────────────────────────────────────────
// EXERCISE: 2D Parallel Reduction
//
// In this exercise, you will implement a 2D parallel reduction to find the 
// total sum of all elements in a large 2D matrix (image).
//
// YOUR TASK:
// Implement the `reduce_2d_kernel`. 
// 1. You are given a 2D grid of 2D thread blocks (e.g., 16x16 threads per block).
// 2. Each thread should load its corresponding matrix element into a 2D 
//    `__shared__` memory array. Be careful if the matrix dimensions are not 
//    perfect multiples of the block size! (Pad with 0.0f).
// 3. Perform a tree-based reduction within the shared memory tile. 
//      Hint: You can reduce along the X-axis first, and then the Y-axis.
//      Or, you can flatten your thread index to 1D 
//      (`int tid = threadIdx.y * blockDim.x + threadIdx.x`)
//      and perform the standard 1D tree reduction you learned earlier!
// 4. Have thread (0,0) of each block safely add its block's partial sum 
//    to the global `total_sum` using `atomicAdd`.
// ─────────────────────────────────────────────────────────────────────────────
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_CUDA_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// ============================================================================
// TODO: Implement the 2D Reduction Kernel
// ============================================================================
__global__ void reduce_2d_kernel(const float *matrix, float *total_sum, int width, int height) {
    // 1. Allocate a 2D shared memory array
    // __shared__ float sdata[BLOCK_DIM_Y][BLOCK_DIM_X];
    
    // 2. Calculate the global (x, y) coordinates for this thread
    int global_x = 0; // REPLACE ME!
    int global_y = 0; // REPLACE ME!
    
    // 3. Load the data into shared memory. 
    // If the thread is outside the matrix bounds, load 0.0f into shared memory!
    
    // 4. Synchronize threads in the block to ensure the tile is fully loaded.
    
    // 5. Perform the tree-based parallel reduction within shared memory.
    
    // 6. Have a single thread (e.g., threadIdx.x == 0 && threadIdx.y == 0) 
    // atomically add the block's final sum into the global `total_sum` variable.
}

// ============================================================================
// HOST CODE (Provided. Do NOT modify unless experimenting)
// ============================================================================
void verify_solution(const std::vector<float>& h_matrix, float gpu_result, int width, int height) {
    std::cout << "Computing CPU reference sum... " << std::flush;
    double cpu_result = 0.0; // Using double to minimize accumulated fp32 precision error natively
    for (int i = 0; i < width * height; ++i) {
        cpu_result += h_matrix[i];
    }
    std::cout << "Done.\n";

    // Allow for a relatively large floating point tolerance due to the massive sum
    // GPU's non-deterministic parallel reduction tree yields radically different 
    // floating-point summation trees compared to CPU's strict linear serial accumulation.
    float diff = std::abs((float)cpu_result - gpu_result);
    float rel_diff = diff / std::abs((float)cpu_result);

    if (rel_diff < 1e-3) {
        std::cout << "SUCCESS! GPU Result matches CPU Result within tolerance.\n";
        std::cout << "  GPU Result: " << gpu_result << "\n";
        std::cout << "  CPU Result: " << (float)cpu_result << "\n";
    } else {
        std::cout << "FAILED! GPU Result does not match CPU Result.\n";
        std::cout << "  Expected: " << (float)cpu_result << "\n";
        std::cout << "  Got:      " << gpu_result << "\n";
        std::cout << "  Rel Diff: " << rel_diff << "\n";
    }
}

int main() {
    int width = 8192;
    int height = 8192;
    int N = width * height; // ~67 million elements
    size_t size = N * sizeof(float);

    std::cout << "2D Matrix Size: " << width << " x " << height << " (" << N << " elements)\n";

    // Allocate host memory
    std::vector<float> h_matrix(N);

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_matrix, *d_total_sum;
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_total_sum, sizeof(float)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, h_matrix.data(), size, cudaMemcpyHostToDevice));
    
    // Initialize the total sum result to strictly zero
    float zero = 0.0f;
    CHECK_CUDA_ERROR(cudaMemcpy(d_total_sum, &zero, sizeof(float), cudaMemcpyHostToDevice));

    // Setup kernel execution configuration
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, 
                  (height + block_dim.y - 1) / block_dim.y);

    // Benchmark the kernel using CUDA Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Launching Kernel...\n";
    cudaEventRecord(start);
    
    // ----------------------------------------------------
    // KERNEL LAUNCH
    // ----------------------------------------------------
    reduce_2d_kernel<<<grid_dim, block_dim>>>(d_matrix, d_total_sum, width, height);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Wait for kernel to finish
    
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel Execution Time: " << ms << " ms\n\n";

    // Copy result back to host
    float h_total_sum;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_total_sum, d_total_sum, sizeof(float), cudaMemcpyDeviceToHost));

    // Verify
    verify_solution(h_matrix, h_total_sum, width, height);

    // Free memory
    cudaFree(d_matrix); cudaFree(d_total_sum);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
