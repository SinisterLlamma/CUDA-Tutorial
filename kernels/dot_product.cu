/*
 * Vector Dot-Product Kernel
 *
 * Each thread block handles a chunk of the two input vectors, multiplies
 * element-wise into shared memory, performs a tree-based reduction, and then
 * atomically accumulates its partial result into a single global output scalar.
 *
 * Usage:
 *   - Grid: ceil(N / BLOCK) blocks
 *   - Block: BLOCK threads  (power of 2, e.g. 256)
 *   - Shared memory: BLOCK * sizeof(float) bytes per block
 */
extern "C" __global__ void dot_product(const float *A, const float *B,
                                       float *result, int N) {
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 1. Element-wise multiply into shared memory
  sdata[tid] = (idx < N) ? A[idx] * B[idx] : 0.0f;
  __syncthreads();

  // 2. Tree-based reduction within the block
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  // 3. Thread 0 of each block atomically adds its partial sum to global result
  if (tid == 0) atomicAdd(result, sdata[0]);
}
