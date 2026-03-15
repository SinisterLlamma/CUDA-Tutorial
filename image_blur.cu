#include <iostream>
#include <vector>
#include <algorithm>

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
// Kernel: 2D Image Blur (Box Filter)
// Demonstrates 2D grids, blocks, and thread indexing
// -----------------------------------------------------------------------------
__global__ void image_blur(const float *d_in, float *d_out, int width, int height) {
  // Compute 2D grid coordinates
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    float sum = 0.0f;
    int count = 0;

    // Apply 3x3 box blur filter
    for (int r = -1; r <= 1; ++r) {
      for (int c = -1; c <= 1; ++c) {
        int targetRow = row + r;
        int targetCol = col + c;

        // Boundary check
        if (targetRow >= 0 && targetRow < height && 
            targetCol >= 0 && targetCol < width) {
          sum += d_in[targetRow * width + targetCol];
          count++;
        }
      }
    }

    // Write the average to the output pixel
    d_out[row * width + col] = sum / (float)count;
  }
}

int main() {
  int width = 1024;
  int height = 1024;
  size_t bytes = width * height * sizeof(float);

  std::vector<float> h_in(width * height);
  std::vector<float> h_out(width * height, 0.0f);

  // Initialize with a simple horizontal gradient
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      h_in[r * width + c] = (float)(c % 256);
    }
  }

  // Allocate GPU Memory
  float *d_in, *d_out;
  CHECK_CUDA_ERROR(cudaMalloc(&d_in, bytes));
  CHECK_CUDA_ERROR(cudaMalloc(&d_out, bytes));

  // Copy data
  CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  // Configure 2D Grid and Blocks
  dim3 block_dim(16, 16);
  dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, 
                (height + block_dim.y - 1) / block_dim.y);

  std::cout << "Launching 2D Blur Kernel: \n"
            << "  Grid : (" << grid_dim.x << ", " << grid_dim.y << ")\n"
            << "  Block: (" << block_dim.x << ", " << block_dim.y << ")\n";

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  
  // Launch kernel
  image_blur<<<grid_dim, block_dim>>>(d_in, d_out, width, height);
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

  // Retrieve result
  CHECK_CUDA_ERROR(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  // Verify simple blur correctness (top-left corner)
  // Corner at (0,0) originally 0. Neighbors: (0,1)=1, (1,0)=0, (1,1)=1.
  // Sum = 0+1+0+1 = 2. Count = 4. Expected Avg = 0.5.
  float expected_0_0 = 0.5f;
  std::cout << "Expected top-left corner blur: " << expected_0_0 << std::endl;
  std::cout << "Actual top-left corner blur  : " << h_out[0] << std::endl;

  if (std::abs(h_out[0] - expected_0_0) < 1e-5) {
      std::cout << "Success! Blur filter seems correct." << std::endl;
  } else {
      std::cerr << "Mismatch on basic verification." << std::endl;
  }

  // Cleanup
  cudaFree(d_in);
  cudaFree(d_out);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
