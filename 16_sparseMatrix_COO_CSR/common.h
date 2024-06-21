#ifndef COMMON_H
#define COMMON_H

struct COOMatrix
{
    unsigned int numRows;     // num of rows in the matrix
    unsigned int numCols;     // num of cols in the matrix
    unsigned int numNonzeros; // num of non zero elements in the matrix

    unsigned int *rowIds; // array of row indices for the non zero elements
    unsigned int *colIds; // array of col indices for the non zero elements
    float *values;
};

void spmv_coo_gpu(COOMatrix cooMatrix, float *inVector, float *outVector);

// class COOMatrix
// {

// public:
//     unsigned int numRows;     // num of rows in the matrix
//     unsigned int numCols;     // num of cols in the matrix
//     unsigned int numNonzeros; // num of non zero elements in the matrix

//     unsigned int *rowIds; // array of row indices for the non zero elements
//     unsigned int *colIds; // array of col indices for the non zero elements
//     float *values;        // array of values for non-zero elements
//     COOMatrix();
//     ~COOMatrix();

//     COOMatrix::COOMatrix() : numRows(0), numCols(0), numNonzeros(0), rowIds(nullptr), colIds(nullptr), values(nullptr){};
//     COOMatrix::~COOMatrix()
//     {
//         // if needed, hndle mem dealloc
//         // normall the GPU memeory allocation/dealloc shuol dbe handled where its needed
//         if (rowIds)
//             delete[] rowIds;
//         if (colIds)
//             delete[] colIds;
//         if (values)
//             delete[] values;
//     }
// };

// struct COOMatrix
// {
//     unsigned int numRows;     // num of rows in the matrix
//     unsigned int numCols;     // num of cols in the matrix
//     unsigned int numNonzeros; // num of non zero elements in the matrix

//     unsigned int *rowIds; // array of row indices for the non zero elements
//     unsigned int *colIds; // array of col indices for the non zero elements
//     float* values;
//     COOMatrix() : numRows(0), numCols(0), numNonzeros(0), rowIds(nullptr), colIds(nullptr), values(nullptr) {} float *values;        // array of values for non-zero elements
// };

#endif // COMMON_H