/*
 * SGEMM Kernels — Progressively Optimised
 *
 * All kernels compute  C = alpha * A * B + beta * C
 * A is M×K, B is K×N, C is M×N  (row-major storage).
 *
 * Optimisation ladder (from SGEMM_CUDA / Simon Boehm's blog):
 *
 *  Kernel 1  sgemm_naive            — 1 thread per C element, uncoalesced
 *  Kernel 2  sgemm_gmem_coalesce    — thread remapping for coalesced GMEM
 *  Kernel 3  sgemm_smem_tiled       — shared-memory tile caching (removes GMEM traffic)
 *  Kernel 4  sgemm_1d_blocktiling   — each thread computes TM elements  (register reuse)
 *  Kernel 5  sgemm_2d_blocktiling   — each thread computes TM×TN elements (outer product)
 */

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Naive — 1 thread = 1 element of C
//   Threads in the same warp read non-contiguous rows of A  → uncoalesced.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void sgemm_naive(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x; // row of C
  const int y = blockIdx.y * blockDim.y + threadIdx.y; // col of C

  if (x < M && y < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Global-Memory Coalescing
//   Re-map thread IDs so adjacent threads in a warp touch adjacent columns.
//   GMEM reads become coalesced (one 128-byte transaction per warp).
//   BLOCKSIZE is passed as a runtime parameter.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void sgemm_gmem_coalesce(int M, int N, int K,
                                               float alpha, const float *A,
                                               const float *B, float beta,
                                               float *C, int BLOCKSIZE) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (cRow < M && cCol < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: Shared-Memory Tiled SGEMM
//   Load TILE_SIZE×TILE_SIZE sub-tiles of A and B into shared memory.
//   Drastically reduces GMEM traffic; bottleneck shifts to SMEM bandwidth.
// ─────────────────────────────────────────────────────────────────────────────
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

extern "C" __global__ void sgemm_smem_tiled(int M, int N, int K, float alpha,
                                            const float *A, const float *B,
                                            float beta, float *C) {
  __shared__ float As[TILE_SIZE * TILE_SIZE];
  __shared__ float Bs[TILE_SIZE * TILE_SIZE];

  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  const uint threadRow = threadIdx.x / TILE_SIZE;
  const uint threadCol = threadIdx.x % TILE_SIZE;

  // Advance pointers to this block's starting tile
  A += cRow * TILE_SIZE * K;
  B += cCol * TILE_SIZE;
  C += cRow * TILE_SIZE * N + cCol * TILE_SIZE;

  float tmp = 0.0f;
  for (int bkIdx = 0; bkIdx < K; bkIdx += TILE_SIZE) {
    // Cooperatively load one tile of A and B
    As[threadRow * TILE_SIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * TILE_SIZE + threadCol] = B[threadRow * N + threadCol];
    __syncthreads();

    A += TILE_SIZE;
    B += TILE_SIZE * N;

    // Compute partial dot-product for this tile
    for (int dotIdx = 0; dotIdx < TILE_SIZE; ++dotIdx) {
      tmp += As[threadRow * TILE_SIZE + dotIdx] *
             Bs[dotIdx * TILE_SIZE + threadCol];
    }
    __syncthreads();
  }
  C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4: 1D Block Tiling — each thread computes TM elements of C
//   A B-value loaded from SMEM is cached in a register and reused TM times.
//   SMEM loads cut by ~TM×, roughly doubling throughput over Kernel 3.
//
//   Template params (compile-time defines):
//     BM — block tile height in M dimension
//     BN — block tile width  in N dimension
//     BK — block tile depth  (K-slice size)
//     TM — number of C elements each thread computes (along M)
// ─────────────────────────────────────────────────────────────────────────────
#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 8
#endif
#ifndef TM
#define TM 8
#endif

extern "C" __global__ void sgemm_1d_blocktiling(int M, int N, int K,
                                                float alpha, const float *A,
                                                const float *B, float beta,
                                                float *C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const uint threadCol = threadIdx.x % BN;
  const uint threadRow = threadIdx.x / BN;

  // Indices for loading tiles from GMEM → SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;

  // Advance to this block's starting position
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  float threadResults[TM] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Load one tile of A and B into SMEM
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    // Each dot-index: cache one B value in a register, reuse TM times
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float Btmp = Bs[dotIdx * BN + threadCol]; // register cache
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
      }
    }
    __syncthreads();
  }

  // Write TM results to global memory
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta  * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5: 2D Block Tiling — each thread computes TM×TN elements of C
//   Thread loads TM values of A and TN values of B into registers, then
//   performs an outer-product entirely in registers.
//   Arithmetic intensity skyrockets → kernel transitions to compute-bound.
//
//   Additional compile-time define needed:
//     TN — number of C elements per thread along N
// ─────────────────────────────────────────────────────────────────────────────
#ifndef TN
#define TN 8
#endif

// Shared memory dimensions for 2D tiling
#define BM2 128
#define BN2 128
#define BK2 8

extern "C" __global__ void sgemm_2d_blocktiling(int M, int N, int K,
                                                float alpha, const float *A,
                                                const float *B, float beta,
                                                float *C) {
  __shared__ float As2[BM2 * BK2];
  __shared__ float Bs2[BK2 * BN2];

  // Each thread handles a TM×TN sub-tile within the block tile
  const uint threadRow = threadIdx.x / (BN2 / TN);
  const uint threadCol = threadIdx.x % (BN2 / TN);

  // GMEM → SMEM loader indices (strided loading for coalescing)
  const uint strideA = (BM2 * BK2) / (BM2 / TM * BN2 / TN);
  const uint strideB = (BK2 * BN2) / (BM2 / TM * BN2 / TN);
  const uint innerRowA = threadIdx.x / BK2;
  const uint innerColA = threadIdx.x % BK2;
  const uint innerRowB = threadIdx.x / BN2;
  const uint innerColB = threadIdx.x % BN2;

  A += blockIdx.y * BM2 * K;
  B += blockIdx.x * BN2;
  C += blockIdx.y * BM2 * N + blockIdx.x * BN2;

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK2) {
    // Load tile of A — each thread loads one element
    for (uint loadIdx = 0; loadIdx < BM2 * BK2 / (BM2 / TM * BN2 / TN); ++loadIdx) {
      uint row = (threadIdx.x + loadIdx * (BM2 / TM * BN2 / TN)) / BK2;
      uint col = (threadIdx.x + loadIdx * (BM2 / TM * BN2 / TN)) % BK2;
      if (row < BM2 && col < BK2)
        As2[row * BK2 + col] = A[row * K + col];
    }
    // Load tile of B
    for (uint loadIdx = 0; loadIdx < BK2 * BN2 / (BM2 / TM * BN2 / TN); ++loadIdx) {
      uint row = (threadIdx.x + loadIdx * (BM2 / TM * BN2 / TN)) / BN2;
      uint col = (threadIdx.x + loadIdx * (BM2 / TM * BN2 / TN)) % BN2;
      if (row < BK2 && col < BN2)
        Bs2[row * BN2 + col] = B[row * N + col];
    }
    __syncthreads();

    A += BK2;
    B += BK2 * N;

    // Outer product in registers
    for (uint dotIdx = 0; dotIdx < BK2; ++dotIdx) {
      for (uint i = 0; i < TM; ++i)
        regM[i] = As2[(threadRow * TM + i) * BK2 + dotIdx];
      for (uint i = 0; i < TN; ++i)
        regN[i] = Bs2[dotIdx * BN2 + threadCol * TN + i];

      for (uint resM = 0; resM < TM; ++resM)
        for (uint resN = 0; resN < TN; ++resN)
          threadResults[resM * TN + resN] += regM[resM] * regN[resN];
    }
    __syncthreads();
  }

  // Write TM×TN results to global memory
  for (uint resM = 0; resM < TM; ++resM) {
    for (uint resN = 0; resN < TN; ++resN) {
      uint row = threadRow * TM + resM;
      uint col = threadCol * TN + resN;
      C[row * N + col] = alpha * threadResults[resM * TN + resN] +
                         beta  * C[row * N + col];
    }
  }
}
