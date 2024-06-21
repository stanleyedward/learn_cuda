#include "common.h"
__global__ void spmv_csr_kernel(CSRMatrix csrMatrix, float *inVector, float *outVector)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    // boundary check
    if (row < csrMatrix.numRows)
    {   
        float sum = 0.0f;
        for(unsigned int i=csrMatrix.rowPtrs[row]; i < csrMatrix.rowPtrs[row + 1]; ++i){
            unsigned int col = csrMatrix.colIdxs[i];
            float value = csrMatrix.values[i];
            sum += value*inVector[col];
        }
        outVector[row]= sum;
    }
}

void spmv_csr_gpu(CSRMatrix csrMatrix, float *inVector, float *outVector)
{
}