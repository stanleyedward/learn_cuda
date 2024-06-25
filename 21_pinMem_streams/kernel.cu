#include "common.h"
#include "timer.h"

__host__ __device__ float f(float a, float b)
{
    return a + b;
}

// single program multiple data = multiple threads exec the same program on different data
__global__ void vecadd_kernel(float *x, float *y, float *z, int N)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; // get threads global index
    if (i < N)
    {
        z[i] = f(x[i], y[i]);
    }
}

void vecadd_gpu(float *x, float *y, float *z, int N)
{

    Timer timer;
    // Allocate GPU mem
    startTime(&timer);
    float *x_d, *y_d, *z_d;
    cudaMalloc((void **)&x_d, N * sizeof(float));
    cudaMalloc((void **)&y_d, N * sizeof(float));
    cudaMalloc((void **)&z_d, N * sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // copy to the GPU
    startTime(&timer);
    cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time", GREEN);

    // Run the GPU code Vector addition
    // call a GPU kernel function (launch a grid of threads)
    startTime(&timer);
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, z_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Error Handling
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        // ... error handling code
    }
    cudaError_t last_error = cudaGetLastError();

    // Copy from the GPU to the CPU
    startTime(&timer);
    cudaMemcpy(z, z_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to CPU time", GREEN);

    // Deallocate GPU memory
    startTime(&timer);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}

void vecadd_gpu_streams(float *x, float *y, float *z, int N)
{

    Timer timer;
    // Allocate GPU mem
    startTime(&timer);
    float *x_d, *y_d, *z_d;
    cudaMalloc((void **)&x_d, N * sizeof(float));
    cudaMalloc((void **)&y_d, N * sizeof(float));
    cudaMalloc((void **)&z_d, N * sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // setup streams
    unsigned int numStreams = 32;
    cudaStream_t stream[numStreams];
    for (unsigned int s = 0; s < numStreams; ++s)
    {
        cudaStreamCreate(&stream[s]);
    }

    // stream the segments
    unsigned int numSegments = numStreams;
    unsigned int segmentSize = (N + numSegments - 1) / numSegments; // ceiling it as it migh tnot be divisble.
    startTime(&timer);
    for (unsigned int s = 0; s < numSegments; ++s)
    {
        // finding the segment bounds
        unsigned int start = s * segmentSize;
        unsigned int end = (start + segmentSize < N) ? (start + segmentSize) : N;
        unsigned int Nsegment = end - start;
        // copy to the GPU
        // we can use either 
        // cudaMemcpyAsync(x_d + start, x + start, Nsegment * sizeof(float), cudaMemcpyHostToDevice, stream[s]);
        // cudaMemcpyAsync(y_d + start, y + start, Nsegment * sizeof(float), cudaMemcpyHostToDevice, stream[s]);

        cudaMemcpyAsync(&x_d[start], &x[start], Nsegment * sizeof(float), cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpyAsync(&y_d[start], &y[start], Nsegment * sizeof(float), cudaMemcpyHostToDevice, stream[s]);
        // Run the GPU code Vector addition
        // call a GPU kernel function (launch a grid of threads)
        const unsigned int numThreadsPerBlock = 512;
        const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
        vecadd_kernel<<<numBlocks, numThreadsPerBlock, 0, stream[s] >>>(&x_d[start], &y_d[start], &z_d[start], Nsegment);

        // Error Handling
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            // ... error handling code
        }
        cudaError_t last_error = cudaGetLastError();

        // Copy from the GPU to the CPU
        cudaMemcpyAsync(&z[start], &z_d[start], Nsegment * sizeof(float), cudaMemcpyDeviceToHost, stream[s]);
    }
    // we have to remove cudadevicesync to outside as we wanna do them in parallel 
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to /from GPU and Kernel time", GREEN);

    // Deallocate GPU memory
    startTime(&timer);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}