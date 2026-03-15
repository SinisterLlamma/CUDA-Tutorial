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

#define FILTER_RADIUS 1
#define FILTER_SIZE (2 * FILTER_RADIUS + 1) // 3x3 filter

// ============================================================================
// EXERCISE 2: Image Convolution (Edge Detection)
// ============================================================================
// Your task is to apply a 3x3 convolution filter to a 2D image.
// The filter weights are stored in the GPU's extremely fast `__constant__` memory.
//
// For every pixel (row, col) in the input image, the output pixel should be the 
// sum of the element-wise multiplication of the 3x3 filter and the 3x3 window of 
// the input image centered at (row, col).
// ============================================================================

// The filter array lives in __constant__ memory on the GPU, so all threads
// can access it extremely quickly simultaneously.
__constant__ float d_Filter[FILTER_SIZE][FILTER_SIZE];

__global__ void convolution_kernel(const float *d_in, float *d_out, int width, int height) {
    // TODO 1: Calculate the global row and column from threadIdx, blockIdx, and blockDim
    // int row = ...
    // int col = ...

    // TODO 2: Ensure the thread is strictly within the bounds of the image
    // if (row < height && col < width) { ... }

    // TODO 3: Iterate through the 3x3 filter using two nested loops (e.g. from -1 to 1)

    // TODO 4: For each filter position, compute the target image coordinate:
    // int targetRow = row + r;
    // int targetCol = col + c;

    // TODO 5: If the target coordinate is OUT OF BOUNDS of the image (less than 0 
    // or >= max width/height), treat the input pixel value as 0.0f (zero padding).
    // Otherwise, read the pixel value from the 1D input array `d_in`.

    // TODO 6: Multiply the input pixel value by the corresponding filter weight 
    // from the `d_Filter` __constant__ array and accumulate the sum.
    // Note: To match indices with -1 to 1 loops, filter index is [r + FILTER_RADIUS][c + FILTER_RADIUS]

    // TODO 7: Write the final sum to `d_out`
}

// ----------------------------------------------------------------------------
// Host Code (Already complete, do not modify!)
// ----------------------------------------------------------------------------
int main() {
    int width = 1024;
    int height = 1024;
    size_t bytes = width * height * sizeof(float);

    // 3x3 Sobel Edge Detection Filter (Horizontal)
    float h_Filter[FILTER_SIZE][FILTER_SIZE] = {
        { 1.0f,  0.0f, -1.0f},
        { 2.0f,  0.0f, -2.0f},
        { 1.0f,  0.0f, -1.0f}
    };

    // Copy the filter to constant memory BEFORE launching the kernel
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Filter, h_Filter, 
                     FILTER_SIZE * FILTER_SIZE * sizeof(float)));

    std::vector<float> h_in(width * height);
    std::vector<float> h_out(width * height, 0.0f);
    std::vector<float> h_out_ref(width * height, 0.0f);

    // Initialize input image with a procedural pattern
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            h_in[r * width + c] = std::sin(r * 0.1f) + std::cos(c * 0.1f);
        }
    }

    // Compute reference solution on CPU
    std::cout << "Computing CPU reference solution... " << std::flush;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            float sum = 0.0f;
            for (int r = -FILTER_RADIUS; r <= FILTER_RADIUS; ++r) {
                for (int c = -FILTER_RADIUS; c <= FILTER_RADIUS; ++c) {
                    int tRow = row + r;
                    int tCol = col + c;
                    if (tRow >= 0 && tRow < height && tCol >= 0 && tCol < width) {
                        float pixel = h_in[tRow * width + tCol];
                        float weight = h_Filter[r + FILTER_RADIUS][c + FILTER_RADIUS];
                        sum += pixel * weight;
                    }
                }
            }
            h_out_ref[row * width + col] = sum;
        }
    }
    std::cout << "Done.\n";

    // Allocate Device memory
    float *d_in, *d_out;
    CHECK_CUDA_ERROR(cudaMalloc(&d_in, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, bytes));

    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // Block and Grid dimensions
    dim3 block_dim(16, 16);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, 
                  (height + block_dim.y - 1) / block_dim.y);

    std::cout << "Launching Kernel...\n";
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    // ------------------------------------------------------------------------
    // KERNEL LAUNCH
    // ------------------------------------------------------------------------
    convolution_kernel<<<grid_dim, block_dim>>>(d_in, d_out, width, height);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::cout << "Kernel Execution Time: " << ms << " ms\n";

    // Copy back and verify
    CHECK_CUDA_ERROR(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < width * height; ++i) {
        if (std::abs(h_out[i] - h_out_ref[i]) > 1e-4) {
            std::cerr << "Mismatch at index " << i << "! Expected: " << h_out_ref[i] 
                      << " Got: " << h_out[i] << "\n";
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "SUCCESS! Convolution implementation is correct.\n";
    }

    // Cleanup
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
