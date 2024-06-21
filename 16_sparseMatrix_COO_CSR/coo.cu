#include "common.h"

__global__ void spmv_coo_kernel(COOMatrix cooMatrix, float *inVector, float *outVector)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cooMatrix.numNonzeros)
    {
        unsigned int row = cooMatrix.rowIds[i];
        unsigned int col = cooMatrix.colIds[i];
        float value = cooMatrix.values[i];

        atomicAdd(&outVector[row], inVector[col]*value);
    }
}

void spmv_coo_gpu(COOMatrix cooMatrix, float *inVector, float *outVector)
{
    // alloc mem
    COOMatrix cooMatrix_d;
    cooMatrix_d.numRows = cooMatrix.numRows;
    cooMatrix_d.numCols = cooMatrix.numCols;
    cooMatrix_d.numNonzeros = cooMatrix.numNonzeros;
    cudaMalloc((void **)&cooMatrix_d.rowIds, cooMatrix_d.numNonzeros * sizeof(unsigned int));
    cudaMalloc((void **)&cooMatrix_d.colIds, cooMatrix_d.numNonzeros * sizeof(unsigned int));
    cudaMalloc((void **)&cooMatrix_d.values, cooMatrix_d.numNonzeros * sizeof(float));
    float *inVector_d;
    cudaMalloc((void **)&inVector_d, cooMatrix_d.numCols * sizeof(float));
    float *outVector_d;
    cudaMalloc((void **)&outVector_d, cooMatrix_d.numRows * sizeof(float));
    cudaDeviceSynchronize();

    // cpy data to gpu
    cudaMemcpy(cooMatrix_d.rowIds, cooMatrix.rowIds, cooMatrix_d.numNonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cooMatrix_d.colIds, cooMatrix.colIds, cooMatrix_d.numNonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cooMatrix_d.values, cooMatrix.values, cooMatrix_d.numNonzeros * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(inVector_d, inVector, cooMatrix.numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(outVector_d, 0, cooMatrix.numRows * sizeof(float));
    cudaDeviceSynchronize();

    // call the kernel
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (cooMatrix_d.numNonzeros + numThreadsPerBlock - 1) / numThreadsPerBlock;
    spmv_coo_kernel<<<numBlocks, numThreadsPerBlock>>>(cooMatrix_d, inVector_d, outVector_d);
    cudaDeviceSynchronize();

    // copy data to CPU
    cudaMemcpy(outVector, outVector_d, cooMatrix.numRows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Free GPU memory
    cudaFree(cooMatrix_d.rowIds);
    cudaFree(cooMatrix_d.colIds);
    cudaFree(cooMatrix_d.values);
    cudaFree(inVector_d);
    cudaFree(outVector_d);
    cudaDeviceSynchronize();
}