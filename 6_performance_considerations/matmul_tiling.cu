#define TILE_DIM 32 // number of threads in a tile
#define COARSE_FACTOR 4

// the tiled matmul kernel is twice as fast as the regular matmul kernel
__global__ void mm_tiled_kernel(float *A, float *B, float *C, unsigned int N)
{
    // assign noe thread to each element in the output matrix C
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colStart = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;

    // declare shared memory 2D arary
    //  shared memory can be statically and dynamically allocated
    //  extern __shared__ A_s[]; <-- dynamic allocation of shared mem
    //  kernel <<<numBlocks, numThreadsPerBlock, sharedMemPerBlock >>> (...) for exec config in shared mem
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    // sum is floated in the register
    float sum[COARSE_FACTOR];
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
    {
        sum[c] = 0.0f;
    }

    // assign values read from threads into shared mem
    for (unsigned int tile = 0; tile < N / TILE_DIM; ++tile)
    {
        A_s[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_DIM + threadIdx.x];

        for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
        {   
            unsigned int col = colStart + c*TILE_DIM;
            B_s[threadIdx.y][threadIdx.x] = B[(tile * TILE_DIM + threadIdx.y) * N + col];
            __syncthreads();

            for (unsigned int i = 0; i < TILE_DIM; ++i)
            {
                sum[c] += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
            }
            __syncthreads();
        }
    }

    for (unsigned int c = 0; c < COARSE_FACTOR; ++c){
        unsigned int col = colStart + c*TILE_DIM;
        C[row * N + col] = sum[c];
    }
}

void mm_gpu(float *A, float *B, float *C, unsigned int N)
{
    // alloc mem
    float *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, N * N * sizeof(float));
    cudaMalloc((void **)&B_d, N * N * sizeof(float));
    cudaMalloc((void **)&C_d, N * N * sizeof(float));
    cudaDeviceSynchronize();

    // copy mem to gpu
    cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // call kernel
    dim3 numThreadsPerBlock(32, 32, 1);
    dim3 numBlocks((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x/COARSE_FACTOR, (N + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y, 1);
    mm_tiled_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();

    // copy mem to cpu
    cudaMemcpy(C_d, C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free mem
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaDeviceSynchronize();
}

// sequential version run on CPU
// tiling on CPU is also twice as fast as regular just like regular GPU speedup
void mm_cpu(float *A, float *B, float *C, unsigned int N)
{
    for (unsigned int rowTile = 0; rowTile < N / TILE_DIM; ++rowTile)
    {
        for (unsigned int colTile = 0; colTile < N / TILE_DIM; ++colTile)
        {
            for (unsigned int iTile = 0; iTile < N / TILE_DIM; ++iTile)
            {
                for (unsigned int row = rowTile * TILE_DIM; row < (rowTile + 1) * TILE_DIM; ++row)
                {
                    for (unsigned int col = colTile * TILE_DIM; col < (colTile + 1) * TILE_DIM; ++col)
                    {
                        float sum = 0.0f;
                        for (unsigned int i = iTile * TILE_DIM; i < (iTile + 1) * TILE_DIM; ++i)
                        {
                            sum += A[row * N + i] * B[i * N + col];
                        }
                        if (iTile == 0)
                            C[row * N + col] = sum;
                        else
                            C[row * N + col] += sum;
                    }
                }
            }
        }
    }
}