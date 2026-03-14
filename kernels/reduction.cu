/*
 * Parallel Reduction Kernels
 *
 * This file contains two CUDA kernels for parallel reduction (summation):
 *   1. naive_reduction   - each thread handles one element, no shared memory
 *   2. shared_mem_reduction - tree-based reduction using shared memory
 *
 * Both kernels compute partial sums per block; the host must launch a second
 * pass (or use atomicAdd) to accumulate the block results.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Naive Reduction (global memory only, interleaved addressing)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void naive_reduction(const float *g_idata,
                                           float *g_odata, int n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread loads one element into a "local" accumulator via shared memory
  // (we still use __shared__ to keep the interface symmetric, but no tree
  // reduction here – just a sequential scan inside each block)
  extern __shared__ float sdata[];
  sdata[tid] = (idx < n) ? g_idata[idx] : 0.0f;
  __syncthreads();

  // Thread 0 sums all elements in its block sequentially
  if (tid == 0) {
    float sum = 0.0f;
    for (int i = 0; i < blockDim.x; ++i) sum += sdata[i];
    g_odata[blockIdx.x] = sum;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Shared-Memory Tree Reduction
// ─────────────────────────────────────────────────────────────────────────────
// Halves the number of active threads at each step → O(log N) depth.
// Requires blockDim.x to be a power of 2.
extern "C" __global__ void shared_mem_reduction(const float *g_idata,
                                                float *g_odata, int n) {
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load into shared memory; pad with 0 if out of bounds
  sdata[tid] = (idx < n) ? g_idata[idx] : 0.0f;
  __syncthreads();

  // Tree-based reduction: at each step s, threads [0, s) accumulate from [s, 2s)
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // One thread per block writes the partial sum to global memory
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
