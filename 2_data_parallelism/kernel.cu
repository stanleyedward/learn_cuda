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

// int main(int argc, char**argv){

//     cudaDeviceSynchronize();

//     // allocate memory and initialize data
//     Timer timer;
//     unsigned int N = (argc > 1)?(atoi(argv[1])) : (1<<25);
//     float *x = (float*)malloc(N*sizeof(float));
//     float *y = (float*)malloc(N*sizeof(float));
//     float *z = (float*)malloc(N*sizeof(float));
//     for (unsigned int i = 0; i < N; ++i){
//         x[i] = rand();
//         y[i] = rand();
//     }

//     // vector addition on CPU
//     startTime(&timer);
//     vecadd_cpu(x, y, z, N);
//     stopTime(&timer);
//     printElapsedTime(timer, "CPU time", CYAN);

//     // vector addition on GPU
//     startTime(&timer);
//     vecadd_gpu(x, y, z, N);
//     stopTime(&timer);
//     printElapsedTime(timer, "GPU time", DGREEN);

//     // free memory
//     free(x);
//     free(y);
//     free(z);

//     return 0;
// }