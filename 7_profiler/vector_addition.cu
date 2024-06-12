__host__ __device__ float f(float a, float b){
    return a + b;
}

void vecadd_cpu(float *x, float *y, float *z, int N){
    for (unsigned int i = 0; i < N; i++){
        z[i] = f(x[i], y[i]);
    }
}

// single program multiple data = multiple threads exec the same program on different data
__global__ void vecadd_kernel(float *x, float *y, float *z, int N){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; //get threads global index
    if (i < N) {
        z[i] = f(x[i] ,y[i]); 
        }
}
void vecadd_gpu(float *x, float *y, float *z, int N){
    // Allocate GPU mem
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**)&x_d, N*sizeof(float));
    cudaMalloc((void**)&y_d, N*sizeof(float));
    cudaMalloc((void**)&z_d, N*sizeof(float));

    // copy to the GPU
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);
    
    // Run the GPU code Vector addition 
    // call a GPU kernel function (launch a grid of threads)
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    vecadd_kernel <<<numBlocks, numThreadsPerBlock >>> (x_d, y_d, z_d, N); 

    // Error Handling
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
        // ... error handling code
    } 
    cudaError_t last_error = cudaGetLastError();
     
    // Copy from the GPU to the CPU
    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate GPU memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

int main(int argc, char**argv){

    cudaDeviceSynchronize();

    // allocate memory and initialize data
    unsigned int N = (argc > 1)?(atoi(argv[1])) : (1<<25);
    float *x = (float*)malloc(N*sizeof(float));
    float *y = (float*)malloc(N*sizeof(float));
    float *z = (float*)malloc(N*sizeof(float));
    for (unsigned int i = 0; i < N; ++i){
        x[i] = rand();
        y[i] = rand();
    }

    // vector addition on CPU
    vecadd_cpu(x, y, z, N);

    // vector addition on GPU
    vecadd_gpu(x, y, z, N);

    // free memory
    free(x);
    free(y);
    free(z);

    return 0;
}