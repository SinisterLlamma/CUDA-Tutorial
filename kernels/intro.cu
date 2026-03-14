/*
 * CUDA Introduction Kernels
 *
 * Three progressively more interesting kernels to introduce the CUDA
 * programming model to newcomers:
 *
 *   1. hello_cuda          — Every thread prints its own ID (thread indexing demo)
 *   2. vector_add          — The "Hello World" of GPGPU: parallel element-wise addition
 *   3. atomic_counter_race — Demonstrates a data race and how atomicAdd fixes it
 *
 * These map directly to the "Introduction" section of the tutorial notebook.
 */

#include <stdio.h>

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Thread Indexing — "whoami"
//
// Every thread computes its unique global ID from blockIdx and threadIdx and
// writes it to the output array.  This demystifies the indexing arithmetic
// that every other kernel builds on.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void thread_index_demo(int *out, int n) {
  // Global thread index in a 1D grid+block configuration
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = idx;   // just store "who I am"
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Vector Addition  (c[i] = a[i] + b[i])
//
// The canonical first CUDA kernel.
// Each thread is responsible for exactly one element — no loops needed.
// This is the embarrassingly-parallel "single-instruction many-data" pattern.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void vector_add(const float *a, const float *b,
                                      float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3a: Broken counter — DATA RACE (for educational purposes!)
//
// 1,000,000 threads all try to increment the same counter.
// Without atomics, threads clobber each other: read-modify-write is NOT atomic.
// Result will almost certainly be LESS than 1,000,000.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void counter_race(int *counter) {
  // Non-atomic read-modify-write: classic race condition
  int old   = *counter;        // step 1: read
  int newval = old + 1;        // step 2: modify  }  NOT protected!
  *counter  = newval;          // step 3: write   }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3b: Fixed counter — atomicAdd
//
// atomicAdd wraps the read-modify-write into a single indivisible hardware
// instruction.  All 1,000,000 increments are now serialised safely → result
// is always exactly 1,000,000.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void counter_atomic(int *counter) {
  atomicAdd(counter, 1);
}
