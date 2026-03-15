# CUDA Programming Tutorial

Welcome to the basic CUDA programming tutorial! This material will get you started on GPU programming using native CUDA C++.

## 0. Introduction to CUDA Programming

### 0.1 Why GPUs? The Parallelism Argument
A modern CPU has ~8–64 powerful cores optimised for low-latency serial execution. A modern GPU has **thousands of simpler cores** designed for high-throughput data-parallel work.

```
CPU:  [Core 0] [Core 1] ... [Core 63]        ← few, fast, general-purpose
GPU:  [SM 0] [SM 1] ... [SM 109]             ← many SMs, each with 64–128 CUDA cores
     Each SM runs hundreds of threads simultaneously
```
**Rule of thumb:** If you can express your problem as "do the same operation on many independent pieces of data", a GPU will beat a CPU by 10×–100×.

### 0.2 The Nuts and Bolts of Operations
Here are some nuts and bolts of CUDA which will be used countless times.

#### Kernel
A kernel is a special function that runs on your GPU (Graphics Card) instead of your CPU. Think of it like giving instructions to a large team of workers (GPU threads) who can all work at the same time. You mark a kernel with the `__global__` keyword, and it can only return `void`. Example:

```cpp
__global__ void addNumbers(int *a, int *b, int *result) {
    *result = *a + *b;
}
```

#### The CUDA Execution Hierarchy
CUDA organises threads into a three-level hierarchy. This makes managing thousands of threads conceptually easy!

```
Grid
└── Block 0 │ Block 1 │ Block 2 │ ...
    └── Thread 0 │ Thread 1 │ ...
```

1. **Grid**: The entire set of threads launched for a single kernel invocation. Think of it as the overall execution space, organized as a 1D, 2D, or 3D collection of blocks.
2. **Block**: A group of threads that can cooperate and share data quickly through fast shared memory. Blocks can be 1D, 2D, or 3D. Threads within a block can share memory and synchronize with each other.
3. **Thread**: The smallest unit of execution. Each thread executes the kernel code independently and has its own unique ID to know which piece of data to work on.

#### Thread Indexing
Each thread has a unique identifier used to determine its position.

| Variable | Meaning |
|----------|---------|
| `threadIdx` | A 3-component vector (`.x`, `.y`, `.z`) giving thread's position within its block. |
| `blockDim` | A 3-component vector specifying dimensions of the block. |
| `blockIdx` | A 3-component vector giving the block's position within the grid. |
| `gridDim` | A 3-component vector specifying the dimensions of the grid. |

**Calculating Global Thread ID:**
To compute a unique global thread ID (e.g., for accessing a 1D array linearly), we use:
```cpp
// blockIdx.x * blockDim.x gives the starting index of the current block
// threadIdx.x gives the thread's local position within the block
int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x; 
```

#### Helper Types and Launch Configuration
**`dim3`**
A simple way to specify 3D dimensions for grid and block sizes.
```cpp
dim3 blockSize(16, 16, 1);  // 16x16x1 threads per block
dim3 gridSize(8, 8, 1);     // 8x8x1 blocks in grid
```

**`<<<gridSize, blockSize>>>`**
Special syntax used to configure and launch kernels from the CPU (Host) onto the GPU (Device).
```cpp
// Kernel launch
myKernel<<<gridSize, blockSize>>>(a, b, result);
```

### 0.3 Memory Management API
To manage arrays on the GPU, CUDA provides analogues to standard C memory tools:

- **`cudaMalloc`**: Allocates memory on the GPU. (Similar to `malloc`).
- **`cudaMemcpy`**: Copies memory. It handles:
  - `cudaMemcpyHostToDevice` (Copy from CPU array to GPU array)
  - `cudaMemcpyDeviceToHost` (Copy from GPU array back to CPU)
  - `cudaMemcpyDeviceToDevice` (Directly copy between GPU locations)
- **`cudaFree`**: Frees memory on the GPU.
- **`cudaDeviceSynchronize()`**: By default, CPU and GPU work asynchronously to overlap work. This command forces the CPU to stop and wait until all GPU operations are fully complete before continuing.

### 0.4 The Memory Hierarchy
CUDA provides several types of memory, each with crucially different behavior and speeds:

| Memory | Scope | Speed | Size | Keyword | Description |
|--------|-------|-------|------|---------|-------------|
| **Register** | Per thread | ~30 TB/s | ~255 per thread | local variables | private, ultra-fast chip memory for things like loop counters. |
| **Shared Memory** | Per block | ~10 TB/s | 48–164 KiB/SM | `__shared__` | Fast cache-like memory blocks can use to cooperate and share data. |
| **L2 Cache** | Chip-wide | ~3 TB/s | ~40 MiB | automatic | Hardware cache for global memory. |
| **Global Memory** | All threads | ~1.5 TB/s | GiBs | `cudaMalloc` | Slowest but largest. The main RAM of the graphics card. |
| **Constant Memory** | All threads | Fast (cached) | ~64 KiB | `__constant__` | Read-only. Great for broadcast parameters that don't change. |
| **Local Memory** | Per thread | Slow (Global) | N/A | spilled | Spilled off-chip variables when registers run out. Avoid if possible! |

**The golden rule:** Keep data in registers or shared memory as long as possible. Continually reading and writing to Global memory is the bottleneck for most kernels.

### 0.5 Putting It All Together
A simple array addition example bringing kernel definition, indexing, memory macros, and launching together:

```cpp
// 1. Definition (runs on GPU)
__global__ void addArrays(int *a, int *b, int *c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

// 2. Launch (called from CPU)
int main() {
    int size = 10000;
    
    // ... Assume cudaMalloc and cudaMemcpy have happened here ...
    
    int threadsPerBlock = 256;
    // Calculate needed blocks (ceiling division)
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;  
    
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);
    
    cudaDeviceSynchronize(); // Wait for finish
    
    return 0;
}
```

---
## 1. Prerequisites & Resources
Before diving into code, here are basic pointers to NVIDIA documentation:
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

Compile the tutorial examples by running:
```bash
make
```

## 2. Hello World: `hello_cuda.cu`
Start by reviewing and running `hello_cuda.cu`:
```bash
./hello_cuda
```

**Concepts Covered:**
- **Host vs Device:** `h_a` (host) vs `d_a` (device).
- **Memory Allocation:** `cudaMalloc` and `cudaMemcpy`.
- **Kernel Definition:** The `__global__` keyword.
- **Grids & Blocks:** Launching `vector_add<<<grid_size, block_size>>>(...)`.

## 3. Parallel Reduction: `parallel_reduction.cu`
Reductions (like sum, min, max) are fundamental parallel patterns.

Run the example:
```bash
./parallel_reduction
```

**Concepts Covered:**
- **Naive Algorithm:** Threads in a block compute their assigned partial sum efficiently, but thread 0 sequentially adds everything in the block.
- **Efficient Algorithm:** The "Tree" reduction. Threads drop off by half at each step: `sdata[tid] += sdata[tid + s]`. This yields O(log N) depth complexity per block.
- **Shared Memory:** `__shared__` memory is much faster than global memory and allows threads *within the same block* to cooperate.
- **Synchronization:** `__syncthreads()` prevents race conditions by ensuring all threads reach the same point before proceeding.



## 4. 2D Grids & Blocks: `image_blur.cu`
GPUs excel at 2D problems like images and matrices. 

Run the example:
```bash
./image_blur
```

**Concepts Covered:**
- **Images as 1D Arrays:** We map 2D coordinates `(row, col)` to a 1D index `row * width + col`.
- **2D Indexing:** 
  ```cpp
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  ```

## 5. Matrix Multiplication Optimizations: `matrix_mul_demo.cu`
## 5. Matrix Multiplication Optimizations: `matrix_mul_demo.cu`

Matrix multiplication is heavily memory-bound and compute-intensive. This demo walks through 5 iterations of optimizing a standard Matrix Multiplication kernel (SGEMM) to show how leveraging CUDA memory hierarchies speeds up execution drastically.

Run the example:
```bash
./matrix_mul_demo
```

### Kernel 1: Naive Implementation
1 thread computes 1 element, reading directly from Global Memory. Threads within the same warp read non-contiguous elements of matrix A. This lack of spatial locality causes many tiny, separate memory transactions, severely choking global memory bandwidth.

![Visualization of Naive Memory Access Pattern](images/naiveK1.png)
![Memory Access Pattern](images/memaccessK1.png)

### Kernel 2: Global Memory Coalescing
To maximize global memory throughput, we re-map thread IDs to ensure consecutive threads in a warp access continuous memory blocks in *both* A and B. The hardware can now group the memory loads (coalesced accesses).

![Coalesced vs. Non-Coalesced Accesses](images/k2.png)

### Kernel 3: Shared Memory (SMEM) Caching
Even with coalescent accesses, fetching from global memory for every math operation is too slow. We utilize Shared Memory (SMEM)—a very fast, on-chip cache. We load small "tiles" (e.g., 32x32 blocks) of A and B from GMEM into SMEM, sync threads, compute partial dot products, and slide to the next tile.

![Block Tiling and Shared Memory Caching stages](images/kernel3.png)
![Roofline Model](images/rooflinek3.png)

### Kernel 4: 1D Blocktiling (Multiple Results per Thread)
To reduce the strain on Shared Memory, we handle more work in thread-local registers. Instead of computing a single element of C, each thread calculates a 1D column (e.g., 8 elements). It caches a value of B in a register and reuses it across 8 distinct calculations with A.

![1D Block Tiling](images/kernell4.png)

### Kernel 5: 2D Blocktiling (Increasing Arithmetic Intensity)
Each thread is now responsible for a 2D grid of elements (e.g., an 8x8 sub-grid of C). The thread loads 8 elements of A and 8 elements of B into registers. By performing an outer product on these registers, it computes 64 results. The kernel transitions from being memory-bound to being completely **compute-bound**!

![2D Thread Tiling outer product](images/kernel5.png)
**Concepts Covered in this Demo:**

### Kernel 1: Naive Implementation
1 thread computes 1 element, reading directly from Global Memory. Threads within the same warp read non-contiguous elements of matrix A. This lack of spatial locality causes many tiny, separate memory transactions, severely choking global memory bandwidth.

![Visualization of Naive Memory Access Pattern](images/naiveK1.png)
![Memory Access Pattern](images/memaccessK1.png)

### Kernel 2: Global Memory Coalescing
To maximize global memory throughput, we re-map thread IDs to ensure consecutive threads in a warp access continuous memory blocks in *both* A and B. The hardware can now group the memory loads (coalesced accesses).

![Coalesced vs. Non-Coalesced Accesses](images/k2.png)

### Kernel 3: Shared Memory (SMEM) Caching
Even with coalescent accesses, fetching from global memory for every math operation is too slow. We utilize Shared Memory (SMEM)—a very fast, on-chip cache. We load small "tiles" (e.g., 32x32 blocks) of A and B from GMEM into SMEM, sync threads, compute partial dot products, and slide to the next tile.

![Block Tiling and Shared Memory Caching stages](images/kernel3.png)
![Roofline Model](images/rooflinek3.png)

### Kernel 4: 1D Blocktiling (Multiple Results per Thread)
To reduce the strain on Shared Memory, we handle more work in thread-local registers. Instead of computing a single element of C, each thread calculates a 1D column (e.g., 8 elements). It caches a value of B in a register and reuses it across 8 distinct calculations with A.

![1D Block Tiling](images/kernell4.png)

### Kernel 5: 2D Blocktiling (Increasing Arithmetic Intensity)
Each thread is now responsible for a 2D grid of elements (e.g., an 8x8 sub-grid of C). The thread loads 8 elements of A and 8 elements of B into registers. By performing an outer product on these registers, it computes 64 results. The kernel transitions from being memory-bound to being completely **compute-bound**!

![2D Thread Tiling outer product](images/kernel5.png)

## 6. Measuring Kernel Time
You will notice we use `cudaEvent_t` in all examples.

**Why not use C++ `std::chrono` directly?**
Kernel launches are *asynchronous*. If you start a CPU timer, launch a kernel, and stop the timer, that simply measures how long the CPU took to *schedule* the kernel, not run it.

**Proper Timing Pattern:**
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);

cudaEventRecord(start);
my_kernel<<<...>>>(...);
cudaEventRecord(stop);

cudaEventSynchronize(stop); // CRITICAL: Wait for GPU to finish
float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
```

## 7. Homework & Practice Challenges
We have provided two heavily-commented skeleton files in your directory where the host-side boilerplate (memory allocation, data copying, CPU baseline checking, and correctness verification) is already fully handled for you. 

Your ONLY job is to write the `__global__` CUDA kernels!

1. **`exercise_2d_reduction.cu` - 2D Parallel Reduction** 
   You will implement a parallel reduction to find the total sum of all elements in a large 2D matrix (image). This exercise challenges you to manage a 2D tile inside `__shared__` memory, correctly perform a tree-based reduction on the 2D tile, and safely accumulate the block sum using `atomicAdd`.
   *To run:* `make exercise_2d_reduction && ./exercise_2d_reduction`

2. **`exercise_convolution.cu` - 2D Image Convolution** 
   Extend your knowledge from `image_blur.cu` to implement a generalized image convolution filter (e.g., a Sobel Edge Detection filter). The filter matrix has already been copied to the highly optimized `__constant__` memory for you. You must handle boundary conditions (zero padding) and correctly accumulate the neighborhood sums.
   *To run:* `make exercise_convolution && ./exercise_convolution`
