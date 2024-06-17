#include "common.h"
#define BLOCK_DIM 1024
#define COARSE_FACTOR 4

__global__ void histogram_kernel(unsigned char *image, unsigned int *bins, unsigned int width, unsigned int height)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ unsigned int histogram_s[NUM_BINS];
    // assign 0 to all px values in the histogram
    if (threadIdx.x < NUM_BINS)
    {
        histogram_s[threadIdx.x] = 0;
    }
    __syncthreads();

    unsigned int bSegment = blockDim.x * blockIdx.x * COARSE_FACTOR;
    unsigned int tSegment = threadIdx.x * COARSE_FACTOR;

    for (unsigned int i = 0; i < COARSE_FACTOR; ++i)
    {
        if (i + bSegment + tSegment < width * height)
        {
            unsigned char b = image[i];
            atomicAdd(&histogram_s[b], 1);
        }
    }
    __syncthreads();

    // commit to global histogram copy
    if (threadIdx.x < NUM_BINS)
    {
        atomicAdd(&bins[threadIdx.x], histogram_s[threadIdx.x]);
    }
}

void histogram_gpu(unsigned char *image, unsigned int *bins, unsigned int width, unsigned int height)
{
    // alloc mem
    unsigned char *image_d;
    unsigned int *bins_d;
    cudaMalloc((void **)&image_d, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&bins_d, NUM_BINS * sizeof(unsigned int));
    cudaDeviceSynchronize();

    // copy data to gpu
    cudaMemcpy(image_d, image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(bins_d, 0, NUM_BINS * sizeof(unsigned int)); // bins start at 0 before incementings
    cudaDeviceSynchronize();

    // call kernel
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numBlocks = (width * height + (COARSE_FACTOR * numThreadsPerBlock) - 1) / (COARSE_FACTOR * numThreadsPerBlock);
    histogram_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);
    cudaDeviceSynchronize();

    // copy data to CPU
    cudaMemcpy(bins, bins_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Free GPU memory
    cudaFree(image_d);
    cudaFree(bins_d);
    cudaDeviceSynchronize();