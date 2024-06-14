#define BLOCK_DIM 8
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
#define c0 2
#define c1 3

__global__ void stencil_kernel(float *in, float *out, unsigned int N)
{
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    // put the input tile in shared memory
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N)
    {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
    }
    __syncthreads();

    // boundary condition
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1)
    {
        if (threadIdx.x >= 1 && threadIdx.x < blockDim.x - 1
         && threadIdx.y >= 1 && threadIdx.y < blockDim.y - 1
         && threadIdx.z >= 1 && threadIdx.z < blockDim.z - 1)
        {
            // identify what output element a threads is resposible for
            //  as c stores multidim array in a 1darray || 3d -> 1d in row major order
            out[i * N * N + j * N + k] = c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
                                         c1 * (in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
                                               in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
                                               in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
                                               in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
                                               in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
                                               in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x]);
        }
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
    // no. threads per block needds to be config according to input tile dim
    dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    // however no. of blocks needed should be enough for all output elementss
    dim3 numBlocks((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    stencil_kernel<<<numBlocks, numThreadsPerBlock>>>(in_d, out_d, N);
    cudaDeviceSynchronize();

    // copy mem to cpu
    cudaMemcpy(out, out_d, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free gpu mem
    cudaFree(in_d);
    cudaFree(out_d);
}