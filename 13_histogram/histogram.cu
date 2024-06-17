#include "common.h"
__global__ void histogram_kernel(unsigned char *image, unsigned int *bins, unsigned int width, unsigned int height)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height){
        unsigned char b = image[i];
        atomicAdd(&bins[b], 1);
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
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (width * height + numThreadsPerBlock - 1) / numThreadsPerBlock;
    histogram_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);
    cudaDeviceSynchronize();

    // copy data to CPU
    cudaMemcpy(bins, bins_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Free GPU memory
    cudaFree(image_d);
    cudaFree(bins_d);
    cudaDeviceSynchronize();