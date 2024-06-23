#include "common.h"
__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int *level, unsigned int *prevFrontier, unsigned int *currFrontier, unsigned int numPrevFrontier, unsigned int *numCurrFrontier, unsigned int currLevel)
{
}

void bfs_gpu(CSRGraph csrGraph, unsigned int srcVertex, unsigned int *level)
{
    // alloc mem
    CSRGraph csrGraph_d;
    csrGraph_d.numVertices = csrGraph.numVertices;
    csrGraph_d.numEdges = csrGraph.numEdges;
    cudaMalloc((void **)&csrGraph_d.srcPtrs, (csrGraph_d.numVertices + 1) * sizeof(unsigned int));
    cudaMalloc((void **)&csrGraph_d.dst, csrGraph_d.numEdges * sizeof(unsigned int));

    unsigned int *level_d;
    cudaMalloc((void **)&level_d, csrGraph_d.numVertices * sizeof(unsigned int));

    // for frontier based approach
    unsigned int *buffer1_d;
    unsigned int *buffer2_d;
    unsigned int *numCurrenFrontier_d;
    cudaMalloc((void **)&buffer1_d, csrGraph_d.numVertices * sizeof(unsigned int));
    cudaMalloc((void **)&buffer2_d, csrGraph_d.numVertices * sizeof(unsigned int));
    cudaMalloc((void **)&numCurrenFrontier_d, sizeof(unsigned int));

    unsigned int *prevFrontier_d = buffer1_d;
    unsigned int *currFrontier_d = buffer2_d;
    cudaDeviceSynchronize();

    // copy data to GPU
    cudaMemcpy(csrGraph_d.srcPtrs, csrGraph.srcPtrs, (csrGraph_d.numVertices + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrGraph_d.dst, csrGraph.dst, csrGraph_d.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
    level[srcVertex] = 0;
    cudaMemcpy(level_d, level, csrGraph_d.numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(prevFrontier_d, &srcVertex, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // run kernel
    unsigned int numPrevFrontier = 1;
    unsigned int numThreadsPerBlock = 256;
    for (unsigned int currLevel = 1; numPrevFrontier > 0; ++currLevel)
    {
        cudaMemset(numCurrenFrontier_d, 0, sizeof(unsigned int));
        unsigned int numBlocks = (numPrevFrontier + numThreadsPerBlock - 1) / numThreadsPerBlock;
        bfs_kernel<<<numBlocks, numThreadsPerBlock>>>(csrGraph_d, level_d, prevFrontier_d, currFrontier_d, numPrevFrontier, numCurrenFrontier_d, currLevel);

        // swap buffers
        unsigned int *tmp = prevFrontier_d;
        prevFrontier_d = currFrontier_d;
        currFrontier_d = tmp;
        cudaMemcpy(&numPrevFrontier, numCurrenFrontier_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    // copy reuslt to CPU
    cudaMemcpy(level, level_d, csrGraph.numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free mem
    cudaFree(csrGraph_d.srcPtrs);
    cudaFree(csrGraph_d.dst);
    cudaFree(level_d);
    cudaFree(buffer1_d);
    cudaFree(buffer2_d);
    cudaFree(numCurrenFrontier_d);
    cudaDeviceSynchronize();
}