#define BLOCK_DIM 1024
#define COARSE_FACTOR 8 // every thread takes 8 elements to scan them sequentially

__global__ void scan_kernel(float *input, float *output, float *partialSums, unsigned int N)
{
    // 2 types of segments block segment and thread segment
    unsigned int bSegment = BLOCK_DIM * COARSE_FACTOR * blockIdx.x;
    __shared__ float buffer_s[BLOCK_DIM * COARSE_FACTOR];
    // to load 8 elements in the block in a coalesced manner
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
    {
        buffer_s[c * BLOCK_DIM + threadIdx.x] = input[bSegment + c * BLOCK_DIM + threadIdx.x];
    }
    __syncthreads();

    // thread scan sequential
    unsigned int tSegment = COARSE_FACTOR * threadIdx.x;
    for (unsigned int c = 1; c < COARSE_FACTOR; ++c)
    {
        buffer_s[tSegment + c] += buffer_s[tSegment + c - 1];
    }
    __syncthreads();

    __shared__ float buffer1_s[BLOCK_DIM];
    __shared__ float buffer2_s[BLOCK_DIM];
    float *inBuffer_s = buffer1_s;
    float *outBuffer_s = buffer2_s;
    // to access the shared memory buffer we use local idx not global idx
    inBuffer_s[threadIdx.x] = buffer_s[tSegment + COARSE_FACTOR - 1];
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

    if (threadIdx.x > 0)
    {
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
        {
            buffer_s[tSegment + c] += inBuffer_s[threadIdx.x - 1];
        }
    }
    if (threadIdx.x == BLOCK_DIM - 1)
    {
        // the last loop iteration swaps value into inbuffer therefore
        partialSums[blockIdx.x] = inBuffer_s[threadIdx.x];
    }
    __syncthreads();
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
    {
        output[bSegment + c * BLOCK_DIM + threadIdx.x] = buffer_s[c * BLOCK_DIM + threadIdx.x];
    }
}

__global__ void add_kernel(float *output, float *partialSums, unsigned int N)
{
    unsigned int bSegment = BLOCK_DIM * COARSE_FACTOR * blockIdx.x;

    if (blockIdx.x > 0)
    {
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
        {

            output[bSegment + c * BLOCK_DIM + threadIdx.x] += partialSums[blockIdx.x - 1];
        }
    }
}

void scan_gpu_d(float *input_d, float *output_d, unsigned int N)
{
    // configurations
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    //since we increased the num of elements per block due to coarsening
    const unsigned int numElementsPerBlock = numThreadsPerBlock*COARSE_FACTOR;
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