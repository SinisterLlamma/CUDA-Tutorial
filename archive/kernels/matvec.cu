/*
 * Matrix-Vector Product Kernels  (y = A * x)
 *
 * Three progressively optimised versions as described in the README:
 *
 *   V1  naive          — one thread per row, all reads from global memory
 *   V2  tiled_x        — collaboratively cache x in shared memory per tile
 *   V3  parallel_rows  — 2D parallelism: all threads in a block work on one row
 *
 * A is M×N (row-major), x is length N, y is length M.
 */

// ─────────────────────────────────────────────────────────────────────────────
// V1: Naive — one thread per output row
//   Bottleneck: x is loaded M times from slow global memory (once per thread).
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void matvec_naive(const float *A, const float *x,
                                        float *y, int M, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M) {
    float dot = 0.0f;
    for (int col = 0; col < N; ++col) {
      dot += A[row * N + col] * x[col];
    }
    y[row] = dot;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// V2: Tiled — cache x in shared memory
//   All threads in a block collaboratively load one tile of x into SMEM,
//   then each thread reuses it for its own row.
//   x is now read once per block per tile instead of once per thread.
// ─────────────────────────────────────────────────────────────────────────────
#ifndef TILE_SIZE
#define TILE_SIZE 256
#endif

extern "C" __global__ void matvec_tiled(const float *A, const float *x,
                                        float *y, int M, int N) {
  __shared__ float x_shared[TILE_SIZE];

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  float dot = 0.0f;

  for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Collaboratively load one tile of x into shared memory
    int x_idx = t * TILE_SIZE + threadIdx.x;
    x_shared[threadIdx.x] = (x_idx < N) ? x[x_idx] : 0.0f;
    __syncthreads();

    // Each thread accumulates using the cached tile
    if (row < M) {
      for (int col = 0; col < TILE_SIZE; ++col) {
        int global_col = t * TILE_SIZE + col;
        if (global_col < N)
          dot += A[row * N + global_col] * x_shared[col];
      }
    }
    __syncthreads();
  }

  if (row < M) y[row] = dot;
}

// ─────────────────────────────────────────────────────────────────────────────
// V3: 2D Parallelism — all threads in a block collaborate on ONE row
//   Launch:  <<<M, BLOCK_COLS>>>   (one block per row)
//   Each thread strides across columns, partial sums are tree-reduced in SMEM.
//   Reduces per-row latency from O(N) → O(N/BLOCK_COLS) + O(log BLOCK_COLS).
// ─────────────────────────────────────────────────────────────────────────────
#ifndef BLOCK_COLS
#define BLOCK_COLS 256
#endif

extern "C" __global__ void matvec_parallel_rows(const float *A, const float *x,
                                                float *y, int M, int N) {
  __shared__ float partial[BLOCK_COLS];

  int row = blockIdx.x;
  int tid = threadIdx.x;

  // Each thread strides across columns
  float sum = 0.0f;
  for (int col = tid; col < N; col += BLOCK_COLS) {
    sum += A[row * N + col] * x[col];
  }
  partial[tid] = sum;
  __syncthreads();

  // Tree-based reduction within the block
  for (unsigned int s = BLOCK_COLS / 2; s > 0; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    __syncthreads();
  }

  // Thread 0 writes the final result
  if (tid == 0) y[row] = partial[0];
}
