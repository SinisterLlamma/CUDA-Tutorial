#include <iostream>
#include <vector>
#include <numeric>
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

// -----------------------------------------------------------------------------
// Kernel: Naive Reduction
// -----------------------------------------------------------------------------
__global__ void naive_reduction(const float *g_idata, float *g_odata, int n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ float sdata[];
  sdata[tid] = (idx < n) ? g_idata[idx] : 0.0f;
  __syncthreads();

  // Thread 0 accumulates sequentially
  if (tid == 0) {
    float sum = 0.0f;
    for (int i = 0; i < blockDim.x; ++i) sum += sdata[i];
    g_odata[blockIdx.x] = sum;
  }
}

// -----------------------------------------------------------------------------
// Kernel: Efficient Shared-Memory Tree Reduction
// -----------------------------------------------------------------------------
__global__ void shared_mem_reduction(const float *g_idata, float *g_odata, int n) {
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < n) ? g_idata[idx] : 0.0f;
  __syncthreads();

  // Tree reduction
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void test_reduction(int N) {
  std::cout << "\nTesting Reduction with N = " << N << std::endl;
  size_t bytes = N * sizeof(float);

  std::vector<float> h_in(N, 1.0f); // Array of 1s, sum should be N.
  
  int block_size = 256;
  int grid_size = (N + block_size - 1) / block_size;
  size_t shared_mem_bytes = block_size * sizeof(float);
  size_t out_bytes = grid_size * sizeof(float);

  std::vector<float> h_out_naive(grid_size, 0.0f);
  std::vector<float> h_out_tree(grid_size, 0.0f);

  float *d_in, *d_out;
  CHECK_CUDA_ERROR(cudaMalloc(&d_in, bytes));
  CHECK_CUDA_ERROR(cudaMalloc(&d_out, out_bytes));

  CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  float time_naive, time_tree;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // --- Run Naive Reduction ---
  cudaEventRecord(start);
  naive_reduction<<<grid_size, block_size, shared_mem_bytes>>>(d_in, d_out, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_naive, start, stop);
  CHECK_CUDA_ERROR(cudaMemcpy(h_out_naive.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

  // --- Run Tree Reduction ---
  cudaEventRecord(start);
  shared_mem_reduction<<<grid_size, block_size, shared_mem_bytes>>>(d_in, d_out, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_tree, start, stop);
  CHECK_CUDA_ERROR(cudaMemcpy(h_out_tree.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

  // CPU sums the block results
  float sum_naive = 0.0f, sum_tree = 0.0f;
  for (int i = 0; i < grid_size; ++i) {
    sum_naive += h_out_naive[i];
    sum_tree += h_out_tree[i];
  }

  float expected = (float)N;
  
  std::cout << "Expected Sum: " << expected << std::endl;
  std::cout << "Naive Sum    : " << sum_naive << " (" << time_naive << " ms)" << std::endl;
  std::cout << "Tree Sum     : " << sum_tree << " (" << time_tree << " ms)" << std::endl;

  cudaFree(d_in);
  cudaFree(d_out);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main() {
  test_reduction(256);
  test_reduction(1 << 23); // ~8M elements
  return 0;
}
