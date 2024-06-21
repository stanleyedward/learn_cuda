#include "common.h"

__global__ void spmv_ell_gpu(ELLMatrix ellMatrix, float *inVector, float *outVector)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < ellMatrix.numRows)
    {
        float sum = 0.0f;
        for (unsigned int iter = 0; iter < ellMatrix.nnzPerRow[row]; ++iter)
        {
            // since its stored in col-major order
            unsigned int i = iter * ellMatrix.numRows + row;
            unsigned int col = ellMatrix.colIdxs[i];
            float value = ellMatrix.values[i];

            sum += inVector[col]*value;
        }
        outVector[row] = sum;
    }
}