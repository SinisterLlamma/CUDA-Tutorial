#include <iostream>
#include <vector>

// Macro for checking CUDA errors
#define CHECK_CUDA_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// -----------------------------------------------------------------------------
// Kernel: Vector Addition
// Demonstrates global memory, block/grid indexing, and registers.
// -----------------------------------------------------------------------------
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
  // Compute global thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < n) {
    // a[i] and b[i] are read from global memory.
    // The addition happens in registers.
    // The result is written to global memory c[i].
    c[i] = a[i] + b[i];
  }
}

int main() {
  int N = 1000000; // 1 Million elements
  size_t bytes = N * sizeof(float);

  // Allocate Host (CPU) Memory
  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);
  std::vector<float> h_c(N, 0.0f);

  // Allocate Device (GPU) Memory
  float *d_a, *d_b, *d_c;
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));
  CHECK_CUDA_ERROR(cudaMalloc(&d_b, bytes));
  CHECK_CUDA_ERROR(cudaMalloc(&d_c, bytes));

  // Copy Data from Host to Device
  CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  // Define Grid and Block dimensions
  int block_size = 256;
  int grid_size = (N + block_size - 1) / block_size;

  std::cout << "Launching Kernel with Grid: " << grid_size 
            << ", Block: " << block_size << std::endl;

  // Setup CUDA Events for Timing
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  // Record start event
  CHECK_CUDA_ERROR(cudaEventRecord(start));

  // Launch Kernel
  vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, N);

  // Record stop event
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop)); // Wait for kernel to finish

  // Calculate elapsed time
  float milliseconds = 0;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  
  // Check for any kernel launch errors
  CHECK_CUDA_ERROR(cudaGetLastError());
  // Wait for all GPU tasks to complete
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

  // Copy Data from Device back to Host
  CHECK_CUDA_ERROR(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  // Verify Result
  bool success = true;
  for (int i = 0; i < N; ++i) {
    if (h_c[i] != 3.0f) {
      std::cerr << "Mismatch at index " << i << ": expected 3.0, got " << h_c[i] << std::endl;
      success = false;
      break;
    }
  }

  if (success) {
    std::cout << "Success! All values computed correctly." << std::endl;
  }

  // Free Device Memory
  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));

  // Cleanup Events
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  return 0;
}
