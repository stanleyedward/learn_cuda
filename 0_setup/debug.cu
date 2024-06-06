#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    y[index] = x[index] + y[index];
  }
}

void checkCudaError(cudaError_t error, const char* msg) {
  if (error != cudaSuccess) {
    std::cerr << "Error: " << msg << ": " << cudaGetErrorString(error) << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main(void)
{
  int N = 1<<20; // 1M elements
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  checkCudaError(cudaMallocManaged(&x, N * sizeof(float)), "cudaMallocManaged x");
  checkCudaError(cudaMallocManaged(&y, N * sizeof(float)), "cudaMallocManaged y");

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Number of threads per block
  int blockSize = 256;

  // Number of blocks in the grid
  int numBlocks = (N + blockSize - 1) / blockSize;

  // Run kernel on 1M elements on the GPU
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Check for any errors launching the kernel
  checkCudaError(cudaGetLastError(), "Kernel launch failed");

  // Wait for GPU to finish before accessing on host
  checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error");

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  checkCudaError(cudaFree(x), "cudaFree x");
  checkCudaError(cudaFree(y), "cudaFree y");
  
  return 0;
}
