// ─────────────────────────────────────────────────────────────────────────────
// SGEMM Matrix Multiplication Optimizations Demo
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

#define TILE_SIZE 16
#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8
#define BM2 128
#define BN2 128
#define BK2 8

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Naive
// ─────────────────────────────────────────────────────────────────────────────
__global__ void sgemm_naive(int M, int N, int K, float alpha,
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
// ─────────────────────────────────────────────────────────────────────────────
__global__ void sgemm_gmem_coalesce(int M, int N, int K,
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
// ─────────────────────────────────────────────────────────────────────────────
__global__ void sgemm_smem_tiled(int M, int N, int K, float alpha,
                                            const float *A, const float *B,
                                            float beta, float *C) {
  __shared__ float As[TILE_SIZE * TILE_SIZE];
  __shared__ float Bs[TILE_SIZE * TILE_SIZE];

  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  const uint threadRow = threadIdx.x / TILE_SIZE;
  const uint threadCol = threadIdx.x % TILE_SIZE;

  A += cRow * TILE_SIZE * K;
  B += cCol * TILE_SIZE;
  C += cRow * TILE_SIZE * N + cCol * TILE_SIZE;

  float tmp = 0.0f;
  for (int bkIdx = 0; bkIdx < K; bkIdx += TILE_SIZE) {
    As[threadRow * TILE_SIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * TILE_SIZE + threadCol] = B[threadRow * N + threadCol];
    __syncthreads();

    A += TILE_SIZE;
    B += TILE_SIZE * N;

    for (int dotIdx = 0; dotIdx < TILE_SIZE; ++dotIdx) {
      tmp += As[threadRow * TILE_SIZE + dotIdx] *
             Bs[dotIdx * TILE_SIZE + threadCol];
    }
    __syncthreads();
  }
  C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4: 1D Block Tiling (Registers)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void sgemm_1d_blocktiling(int M, int N, int K,
                                                float alpha, const float *A,
                                                const float *B, float beta,
                                                float *C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const uint threadCol = threadIdx.x % BN;
  const uint threadRow = threadIdx.x / BN;

  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  float threadResults[TM] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float Btmp = Bs[dotIdx * BN + threadCol]; // register cache
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
      }
    }
    __syncthreads();
  }

  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta  * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5: 2D Block Tiling
// ─────────────────────────────────────────────────────────────────────────────
__global__ void sgemm_2d_blocktiling(int M, int N, int K,
                                                float alpha, const float *A,
                                                const float *B, float beta,
                                                float *C) {
  __shared__ float As2[BM2 * BK2];
  __shared__ float Bs2[BK2 * BN2];

  const uint threadRow = threadIdx.x / (BN2 / TN);
  const uint threadCol = threadIdx.x % (BN2 / TN);

  A += blockIdx.y * BM2 * K;
  B += blockIdx.x * BN2;
  C += blockIdx.y * BM2 * N + blockIdx.x * BN2;

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK2) {
    // Load tile of A
    for (uint loadIdx = 0; loadIdx < BM2 * BK2 / (BM2 / TM * BN2 / TN); ++loadIdx) {
      uint row = (threadIdx.x + loadIdx * (BM2 / TM * BN2 / TN)) / BK2;
      uint col = (threadIdx.x + loadIdx * (BM2 / TM * BN2 / TN)) % BK2;
      if (row < BM2 && col < BK2) As2[row * BK2 + col] = A[row * K + col];
    }
    // Load tile of B
    for (uint loadIdx = 0; loadIdx < BK2 * BN2 / (BM2 / TM * BN2 / TN); ++loadIdx) {
      uint row = (threadIdx.x + loadIdx * (BM2 / TM * BN2 / TN)) / BN2;
      uint col = (threadIdx.x + loadIdx * (BM2 / TM * BN2 / TN)) % BN2;
      if (row < BK2 && col < BN2) Bs2[row * BN2 + col] = B[row * N + col];
    }
    __syncthreads();

    A += BK2;
    B += BK2 * N;

    // Outer product in registers
    for (uint dotIdx = 0; dotIdx < BK2; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) regM[i] = As2[(threadRow * TM + i) * BK2 + dotIdx];
      for (uint i = 0; i < TN; ++i) regN[i] = Bs2[dotIdx * BN2 + threadCol * TN + i];

      for (uint resM = 0; resM < TM; ++resM)
        for (uint resN = 0; resN < TN; ++resN)
          threadResults[resM * TN + resN] += regM[resM] * regN[resN];
    }
    __syncthreads();
  }

  // Write TMxTN results to global memory
  for (uint resM = 0; resM < TM; ++resM) {
    for (uint resN = 0; resN < TN; ++resN) {
      uint row = threadRow * TM + resM;
      uint col = threadCol * TN + resN;
      C[row * N + col] = alpha * threadResults[resM * TN + resN] + beta  * C[row * N + col];
    }
  }
}

// -----------------------------------------------------------------------------
// Benchmark Runner
// -----------------------------------------------------------------------------
float benchmark(const std::string& name, float *d_A, float *d_B, float *d_C, int M, int N, int K, int kernel_id) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    dim3 block_dim, grid_dim;
    int BS = 32;

    cudaEventRecord(start);
    if (kernel_id == 1) {
        block_dim = dim3(BS, BS);
        grid_dim = dim3((M + BS - 1) / BS, (N + BS - 1) / BS);
        sgemm_naive<<<grid_dim, block_dim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    } else if (kernel_id == 2) {
        block_dim = dim3(BS * BS);
        grid_dim = dim3((M + BS - 1) / BS, (N + BS - 1) / BS);
        sgemm_gmem_coalesce<<<grid_dim, block_dim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C, BS);
    } else if (kernel_id == 3) {
        block_dim = dim3(TILE_SIZE * TILE_SIZE);
        grid_dim = dim3((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
        sgemm_smem_tiled<<<grid_dim, block_dim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    } else if (kernel_id == 4) {
        block_dim = dim3(BM / TM * BN);
        grid_dim = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
        sgemm_1d_blocktiling<<<grid_dim, block_dim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    } else if (kernel_id == 5) {
        int threads = (BM2 / TM) * (BN2 / TN);
        block_dim = dim3(threads);
        grid_dim = dim3((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2);
        sgemm_2d_blocktiling<<<grid_dim, block_dim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    double flops = 2.0 * M * N * K;
    double gflops = (flops / (ms / 1000.0)) / 1e9;
    
    std::cout << "[Kernel " << kernel_id << "] " << name 
              << "\n  Time: " << ms << " ms   |   GFLOPS: " << gflops << "\n\n";

    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}

int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;

    std::cout << "========================================================\n"
              << "SGEMM Matrix Multiplication Benchmark (M=" << M << ", N=" << N << ", K=" << K << ")\n"
              << "========================================================\n\n";

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    benchmark("Naive", d_A, d_B, d_C, M, N, K, 1);
    benchmark("Global Memory Coalescing", d_A, d_B, d_C, M, N, K, 2);
    benchmark("Shared Memory Tiling", d_A, d_B, d_C, M, N, K, 3);
    benchmark("1D Block Tiling (Registers)", d_A, d_B, d_C, M, N, K, 4);
    benchmark("2D Block Tiling (High Arithmetic Intensity)", d_A, d_B, d_C, M, N, K, 5);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
