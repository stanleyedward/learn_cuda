#define TILE_DIM 32 // number of threads in a tile

__global__ void mm_kernel(float *A, float *B, float *C, unsigned int N){
    // assign noe thread to each element in the output matrix C
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    //declare shared memory 2D arary
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    //sum is floated in the register
    float sum = 0.0f;

    // assign values read from threads into shared mem
    for(unsigned int tile = 0; tile < N/TILE_DIM; ++tile){
        A_s[threadIdx.y][threadIdx.x] = A[row*N + tile*TILE_DIM + threadIdx.x];
        B_s[threadIdx.y][threadIdx.x] = B[(tile*TILE_DIM + threadIdx.y)*N + col];
        __syncthreads();

        for(unsigned int i = 0; i < TILE_DIM; ++i){
            sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
        }
        __syncthreads();
    }

    C[row*N + col] = sum;

}

void mm_gpu(float *A, float *B, float *C, unsigned int N){
    //alloc mem
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, N*N*sizeof(float));
    cudaMalloc((void**) &B_d, N*N*sizeof(float));
    cudaMalloc((void**) &C_d, N*N*sizeof(float));
    cudaDeviceSynchronize();

    // copy mem to gpu
    cudaMemcpy(A_d, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    //call kernel
    dim3 numThreadsPerBlock(32, 32, 1);
    dim3 numBlocks((N + numThreadsPerBlock.x -1)/numThreadsPerBlock.x, (N + numThreadsPerBlock.y -1)/numThreadsPerBlock.y, 1);
    mm_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();

    //copy mem to cpu
    cudaMemcpy(C_d, C, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free mem
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaDeviceSynchronize();
}
