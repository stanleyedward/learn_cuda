#define BLOCK_DIM 1024
__global__ void scan_kernel(float *input, float *output, float *partialSums, unsigned int N)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float buffer_s[BLOCK_DIM];
    // to access the shared memory buffer we use local idx not global idx
    buffer_s[threadIdx.x] = input[i];
    __syncthreads();
    for (unsigned int stride = 1; stride <= BLOCK_DIM / 2; stride *= 2)
    {
        float v;
        //sync after read
        if (threadIdx.x >= stride)
        {
            v = buffer_s[threadIdx.x - stride];
        }
        __syncthreads();
        //sync after write
        if (threadIdx.x >= stride)
        {
            buffer_s[threadIdx.x] += v;
        }
        __syncthreads();
    }
    if (threadIdx.x == BLOCK_DIM - 1)
    {
        partialSums[blockIdx.x] = buffer_s[threadIdx.x];
    }
    output[i] = buffer_s[threadIdx.x];
}

__global__ void add_kernel(float *output, float *partialSums, unsigned int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0)
    {
        output[i] += partialSums[blockIdx.x - 1];
    }
}

void scan_gpu_d(float *input_d, float *output_d, unsigned int N)
{
    // configurations
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1) / numElementsPerBlock;

    // alloc partial sums
    float *partialSums_d;
    cudaMalloc((void **)&partialSums_d, numBlocks * sizeof(float));
    cudaDeviceSynchronize();

    // call kernel
    scan_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, partialSums_d, N);
    cudaDeviceSynchronize();

    // scan partial sums then add
    if (numBlocks > 1)
    {
        // scan partial sums
        scan_gpu_d(partialSums_d, partialSums_d, numBlocks);

        // add scanned sums
        add_kernel<<<numBlocks, numThreadsPerBlock>>>(output_d, partialSums_d, N);
    }

    // free memory
    cudaFree(partialSums_d);
    cudaDeviceSynchronize();
}