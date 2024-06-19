__host__ __device__ void mergeSequential(float *A, float *B, float *C, unsigned int m, unsigned int n)
{
    unsigned int i = 0; // counter for A
    unsigned int j = 0; // counter for B
    unsigned int k = 0; // counter for C
                        // m is thelength of A and n is the  length of B
    while (i < m && j < n)
    {
        if (A[i] < B[j])
        {
            C[k++] = A[i++]; // C[k++] post increments values;
        }
        else
        {
            C[k++] = B[j++];
        }
    }
    while (i < m)
    {
        C[k++] = A[i++];
    }
    while (j < n)
    {
        C[k++] = B[j++];
    }
}

__device__ unsigned int coRank(float *A, float *B, unsigned int m, unsigned int n, unsigned int k)
{ // get upper and lower bound of i idx in array A

    // unsigned int(lower) - unsigned int (greater) leads to wrapping around giving us a very large number
    // unsigned int iLow = (k - n > 0) ? (k - n) : 0;
    unsigned int iLow = (k > n) ? (k - n) : 0;
    unsigned int iHigh = (m < k) ? m : k;

    // binary search
    while (true)
    {
        unsigned int i = (iLow + iHigh) / 2;
        unsigned int j = k - i;
        // boundary check
        // guess is too high
        if ((i > 0 && j < n) && A[i - 1] > B[j])
        {
            iHigh = i;
        }
        // guess is too low
        else if ((j > 0 && i < m) && B[j - 1] > A[i])
        {
            iLow = i;
        }
        else
        {
            return i;
        }
    }
}

#define ELEM_PER_THREAD 6
#define THREADS_PER_BLOCK 128
#define ELEM_PER_BLOCK (ELEM_PER_THREAD * THREADS_PER_BLOCK)

__global__ void merge_kernel(float *A, float *B, float *C, unsigned int m, unsigned int n)
{
    // identify the block's segments
    unsigned int kBlock = blockIdx.x * ELEM_PER_BLOCK;
    unsigned int kNextBlock = (blockIdx.x < gridDim.x - 1) ? (kBlock + ELEM_PER_BLOCK) : (m + n);

    __shared__ unsigned int iBlock;
    __shared__ unsigned int iNextBlock;
    if (threadIdx.x == 0)//only 1 thread to get iblock for the entire block
    {
        iBlock = coRank(A, B, m, n, kBlock);
        iNextBlock = coRank(A, B, m, n, kNextBlock);
    }
    __syncthreads();
    unsigned int jBlock = kBlock - iBlock;
    unsigned int jNextBlock = kNextBlock - iNextBlock;

    // load blocks segments to shared memory in a coalesced way
    __shared__ float A_s[ELEM_PER_BLOCK];
    unsigned int mBlock = iNextBlock - iBlock;
    for (unsigned int i = threadIdx.x; i < mBlock; i += blockDim.x)
    {
        A_s[i] = A[iBlock + i];
    }
    float *B_s = A_s + mBlock;
    unsigned int nBlock = jNextBlock - jBlock;
    for (unsigned int j = threadIdx.x; j < nBlock; j += blockDim.x)
    {
        B_s[j] = B[jBlock + j];
    }
    __syncthreads();

    // merge block in shared memory
    __shared__ float C_s[ELEM_PER_BLOCK];
    unsigned int k = threadIdx.x * ELEM_PER_THREAD;
    if (k < mBlock + nBlock)
    {
        unsigned int i = coRank(A_s, B_s, mBlock, nBlock, k);
        unsigned int j = k - i;
        unsigned kNext = (k + ELEM_PER_THREAD < mBlock + nBlock) ? (k + ELEM_PER_THREAD) : (nBlock + mBlock);
        unsigned int iNext = coRank(A_s, B_s, mBlock, nBlock, kNext);
        unsigned int jNext = kNext - iNext;
        mergeSequential(&A_s[i], &B_s[j], &C_s[k], iNext - i, jNext - j);
    }
    __syncthreads();

    // write block's segment to global memory in a coalesced way
    for (unsigned int k = threadIdx.x; k < mBlock + nBlock; k += blockDim.x)
    {
        C[kBlock + k] = C_s[k];
    }
}

void merge_gpu(float *A, float *B, float *C, unsigned int m, unsigned int n)
{
    // alloc mem
    float *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, m * sizeof(float));
    cudaMalloc((void **)&B_d, n * sizeof(float));
    cudaMalloc((void **)&C_d, (m + n) * sizeof(float));
    cudaDeviceSynchronize();

    // cpoy data to gpu
    cudaMemcpy(A_d, A, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // call kernel
    unsigned int numBlocks = (m + n + ELEM_PER_BLOCK - 1) / ELEM_PER_BLOCK;
    merge_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(A_d, B_d, C_d, m, n);
    cudaDeviceSynchronize();

    // copy data to cpu
    cudaMemcpy(C, C_d, (m + n) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}