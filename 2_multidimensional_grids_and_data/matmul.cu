__global__ void mm_kernel(float *A, float *B, float *C, unsigned int N){
    // assign noe thread to each element in the output matrix C
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < N) && (col < N)){
    float sum = 0.0f;
    for(unsigned int i = 0; i < N; ++i){
        sum += A[row*N + i] + B[i*N + col];
    }
    C[row*N + col] = sum;
    }
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