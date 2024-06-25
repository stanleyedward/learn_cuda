#include "common.h"
#include "timer.h"
__host__ __device__ float f_cpu(float a, float b)
{
    return a + b;
}
void vecadd_cpu(float *x, float *y, float *z, int N)
{
    for (unsigned int i = 0; i < N; i++)
    {
        z[i] = f_cpu(x[i], y[i]);
    }
}
int main(int argc, char **argv)
{

    cudaDeviceSynchronize();

    // allocate memory and initialize data
    Timer timer;
    unsigned int N = (argc > 1) ? (atoi(argv[1])) : (1 << 25);
    float *x;
    cudaMallocHost((void **)&x, N * sizeof(float));
    float *y;
    cudaMallocHost((void **)&y, N * sizeof(float));
    float *z;
    cudaMallocHost((void **)&z, N * sizeof(float));
    // float *z_gpu;
    // cudaMallocHost((void **)&z_gpu, N * sizeof(float));
    for (unsigned int i = 0; i < N; ++i)
    {
        x[i] = rand();
        y[i] = rand();
    }

    // vector addition on CPU
    startTime(&timer);
    vecadd_cpu(x, y, z, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // // vector addition on GPU
    // startTime(&timer);
    // vecadd_gpu(x, y, z, N);
    // stopTime(&timer);
    // printElapsedTime(timer, "GPU time", DGREEN);

    // vector addition on GPU with streams
    startTime(&timer);
    vecadd_gpu_streams(x, y, z, N);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // free memory
    cudaFreeHost(x);
    cudaFreeHost(y);
    cudaFreeHost(z);
    // cudaFreeHost(z_gpu);

    return 0;
}