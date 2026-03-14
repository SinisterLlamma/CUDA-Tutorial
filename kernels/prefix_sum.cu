/*
 * Parallel Prefix Sum (Scan) Kernels
 *
 * Two classic algorithms, both operate within a single block:
 *   1. hillis_steele_scan  - Step-efficient (O(N log N) work) but simple / parallel
 *   2. blelloch_scan       - Work-efficient (O(N) work) using up-sweep + down-sweep
 *
 * For arrays larger than blockDim.x, the host must implement multi-level scan.
 */

#define MAX_BLOCK 1024

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Hillis-Steele Inclusive Scan
// ─────────────────────────────────────────────────────────────────────────────
// At each step d, every thread i adds element at (i - 2^(d-1)) to itself.
// Simple and highly parallel but NOT work-efficient.
extern "C" __global__ void hillis_steele_scan(const float *g_idata,
                                              float *g_odata, int n) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  // Load input
  sdata[tid] = (idx < n) ? g_idata[idx] : 0.0f;
  __syncthreads();

  // Hillis-Steele iterative scan
  for (int d = 1; d < blockDim.x; d *= 2) {
    float val = 0.0f;
    if (tid >= d) val = sdata[tid - d];
    __syncthreads();
    sdata[tid] += val;
    __syncthreads();
  }

  if (idx < n) g_odata[idx] = sdata[tid];
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Blelloch Work-Efficient Exclusive Scan
// ─────────────────────────────────────────────────────────────────────────────
// Phase 1 (up-sweep / reduce): build partial-sum tree bottom-up.
// Phase 2 (down-sweep):        zero the root, propagate sums top-down.
// Result is an exclusive prefix sum: g_odata[0] = 0, g_odata[i] = sum(g_idata[0..i-1]).
extern "C" __global__ void blelloch_scan(const float *g_idata, float *g_odata,
                                         int n) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  // Load two elements per thread (double the logical N)
  sdata[2 * tid]     = (2 * idx < n)     ? g_idata[2 * idx]     : 0.0f;
  sdata[2 * tid + 1] = (2 * idx + 1 < n) ? g_idata[2 * idx + 1] : 0.0f;
  __syncthreads();

  int m = blockDim.x * 2; // effective array size in shared mem

  // --- Phase 1: Up-Sweep (Reduce) ---
  for (int d = 1; d < m; d *= 2) {
    int index = (tid + 1) * d * 2 - 1;
    if (index < m) sdata[index] += sdata[index - d];
    __syncthreads();
  }

  // Clear the last element (set identity)
  if (tid == 0) sdata[m - 1] = 0.0f;
  __syncthreads();

  // --- Phase 2: Down-Sweep ---
  for (int d = m / 2; d > 0; d /= 2) {
    int index = (tid + 1) * d * 2 - 1;
    if (index < m) {
      float tmp          = sdata[index - d];
      sdata[index - d]   = sdata[index];
      sdata[index]      += tmp;
    }
    __syncthreads();
  }

  // Write results
  if (2 * idx < n)     g_odata[2 * idx]     = sdata[2 * tid];
  if (2 * idx + 1 < n) g_odata[2 * idx + 1] = sdata[2 * tid + 1];
}
