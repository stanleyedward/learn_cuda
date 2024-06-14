#define BLOCK_DIM 1024

__global__ void reduce_kernel(float *input, float *partialSums, unsigned int N)
{
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + threadIdx.x;
    for (unsigned int stride = BLOCK_DIM; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    // every thread will have a partial sum at threadIdx 0 in the block, therefore have thread 0 store its result in the partial sum array.
    if (threadIdx.x == 0)
    {
        partialSums[blockIdx.x] = input[i];
    }
}

float reduce_gpu(float *input, unsigned int N)
{
    // alloc amem
    float *input_d;
    cudaMalloc((void **)&input_d, N * sizeof(float));
    cudaDeviceSynchronize();

    // copy data to GPU
    cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // alloc partial sums
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2 * numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1) / numElementsPerBlock;
    float *partialSums = (float *)malloc(numBlocks * sizeof(float));
    float *partialSums_d;
    cudaMalloc((void **)&partialSums_d, numBlocks * sizeof(float));
    cudaDeviceSynchronize();

    // call kernel
    reduce_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, partialSums_d, N);
    cudaDeviceSynchronize();

    // copy data from GPU
    cudaMemcpy(partialSums, partialSums_d, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // reduce partial sums on CPU
    float sum = 0.0f;
    for (unsigned int i = 0; i < numBlocks; ++i)
    {
        sum += partialSums[i];
    }

    // free mem
    cudaFree(input_d);
    cudaFree(partialSums_d);
    free(partialSums);
    cudaDeviceSynchronize();
}