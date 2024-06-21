struct  ELLMatrix 
{
    unsigned int numRows;
    unsigned int numCols;
    unsigned int maxNNZPerRow; // to get padding required to get the dim of the 2d array that were gonna flatten
    unsigned int* nnzPerRow; //
    unsigned int* colIdxs;
    float* values;
};

void spmv_ell_gpu(ELLMatrix ellMatrix, float* inVector, float* outVector);
