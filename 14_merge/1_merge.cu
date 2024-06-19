__device__ void mergeSequential(float *A, float *B, float *C, unsigned int m, unsigned int n)
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
    // identify k for each thread
    unsigned int k = (blockDim.x * blockIdx.x + threadIdx.x) * ELEM_PER_THREAD;
    if (k < m + n) // boundary cond.
    {
        unsigned int i = coRank(A, B, m, n, k);
        unsigned int j = k - 1;

        // get size of i and j segment so we can sequentially merge
        unsigned int kNext = (k + ELEM_PER_THREAD) < (m + n) ? (k + ELEM_PER_THREAD) : (m + n);
        unsigned int iNext = coRank(A, B, m, n, kNext);
        unsigned int jNext = kNext - iNext;

        //merge
        mergeSequential(&A[i], &B[j], &C[k], (iNext - i), (jNext - j));
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