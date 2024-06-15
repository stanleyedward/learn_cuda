#define BLOCK_DIM 1024

void scan_cpu(float *input, float *output, unsigned int N)
{
    output[0] = 0.0f;
    for (unsigned int i = 1; i < N; ++i)
    {
        output[i] = output[i - 1] + input[i - 1];
    }
}
__global__ void scan_kernel(float *input, float *output, float *partialSums, unsigned int N)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float buffer1_s[BLOCK_DIM];
    __shared__ float buffer2_s[BLOCK_DIM];
    float *inBuffer_s = buffer1_s;
    float *outBuffer_s = buffer2_s;
    // to access the shared memory buffer we use local idx not global idx
    if (threadIdx.x == 0)
    {
        inBuffer_s[threadIdx.x] = 0.0f;
    }
    else
    {
        inBuffer_s[threadIdx.x] = input[i - 1];
    }
    __syncthreads();
    for (unsigned int stride = 1; stride <= BLOCK_DIM / 2; stride *= 2)
    {
        if (threadIdx.x >= stride)
        {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        }
        else
        { // copy the values from inavtivethreads
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();
        // in the next iteration we need to swap output and input buffer
        float *temp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = temp;
    }
    if (threadIdx.x == BLOCK_DIM - 1)
    {
        // the last loop iteration swaps value into inbuffer therefore
        partialSums[blockIdx.x] = inBuffer_s[threadIdx.x] + input[i];
    }
    output[i] = inBuffer_s[threadIdx.x];
}

__global__ void   add_kernel(float *output, float *partialSums, unsigned int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0)
    {
        output[i] += partialSums[blockIdx.x];
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