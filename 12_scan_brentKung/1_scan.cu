#define BLOCK_DIM 1024
__global__ void scan_kernel(float *input, float *output, float *partialSums, unsigned int N)
{
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    __shared__ float buffer_s[BLOCK_DIM * 2];
    // load data in 2 parts so its memory coalesced
    // every thread resposible for 2 elements
    buffer_s[threadIdx.x] = input[segment + threadIdx.x];
    buffer_s[threadIdx.x + BLOCK_DIM] = input[segment + threadIdx.x + BLOCK_DIM];
    __syncthreads();

    // reduction step
    for (unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2)
    {
        unsigned int i = (threadIdx.x + 1) * 2 * stride - 1;
        if (i < BLOCK_DIM * 2)
        {
            buffer_s[i] = buffer_s[i] + buffer_s[i - stride];
        }
        __syncthreads();
    }

    // post-rediction step
    for (unsigned int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2)
    {
        unsigned int i = (threadIdx.x + 1) * 2 * stride - 1;
        if ((i + stride) < BLOCK_DIM * 2)
        {
            buffer_s[i + stride] = buffer_s[i + stride] + buffer_s[i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        partialSums[blockIdx.x] = buffer_s[2 * BLOCK_DIM - 1];
    }
    // each thread stores 2 output values
    output[segment + threadIdx.x] = buffer_s[threadIdx.x];
    output[segment + threadIdx.x + BLOCK_DIM] = buffer_s[threadIdx.x + BLOCK_DIM];
}

__global__ void add_kernel(float *output, float *partialSums, unsigned int N)
{
    unsigned int segment = 2 * blockIdx.x * blockDim.x;
    if (blockIdx.x > 0)
    {
        output[segment + threadIdx.x] += partialSums[blockIdx.x - 1];
        output[segment + threadIdx.x + BLOCK_DIM] += partialSums[blockIdx.x - 1];
    }
}

void scan_gpu_d(float *input_d, float *output_d, unsigned int N)
{
    // configurations
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2 * numThreadsPerBlock;
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