#define BLOCK_DIM 8
#define c0 2
#define c1 3
__global__ void stencil_kernel(float *in, float *out, unsigned int N)
{
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    // boundary condition
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1)
    {
        // as c stores multidim array in a 1darray || 3d -> 1d in row major order
        out[i * N * N + j * N + k] = c0 * in[i * N * N + j * N + k] +
                                     c1 * (in[i * N * N + j * N + (k - 1)] +
                                           in[i * N * N + j * N + (k + 1)] +
                                           in[i * N * N + (j - 1) * N + k] +
                                           in[i * N * N + (j + 1) * N + k] +
                                           in[(i - 1) * N * N + j * N + k] +
                                           in[(i + 1) * N * N + j * N + k]);
    } 
}

void stencil_gpu(float *in, float *out, unsigned int N)
{
    // alloc mem
    float *in_d, *out_d;
    cudaMalloc((void **)&in_d, N * N * N * sizeof(float));
    cudaMalloc((void **)&out_d, N * N * N * sizeof(float));
    cudaDeviceSynchronize();

    // move mem to gpu
    cudaMemcpy(in_d, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // launch the kernel
    dim3 numThreadsPerBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 numBlocks((N + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM);
    stencil_kernel<<<numBlocks, numThreadsPerBlock>>>(in_d, out_d, N);
    cudaDeviceSynchronize();

    // copy mem to cpu
    cudaMemcpy(out, out_d, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free gpu mem
    cudaFree(in_d);
    cudaFree(out_d);
}