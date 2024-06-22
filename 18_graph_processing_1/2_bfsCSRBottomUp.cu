#include "common.h"
__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int *level, unsigned int *newVertexVisited, unsigned int currLevel)
{
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < csrGraph.numVertices)
    {
        if (level[vertex] == UINT_MAX)
        {
            for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge)
            {
                unsigned int neighbour = csrGraph.dst[edge];
                if (level[neighbour] == currLevel - 1)
                {
                    level[vertex] = currLevel;
                    *newVertexVisited = 1;
                    break; // exit loop early if neighbour as been found
                }
            }
        }
    }
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
    unsigned int *newVertexVisited_d;
    cudaMalloc((void **)&newVertexVisited_d, sizeof(unsigned int));
    cudaDeviceSynchronize();

    // copy data to GPU
    cudaMemcpy(csrGraph_d.srcPtrs, csrGraph.srcPtrs, (csrGraph_d.numVertices + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrGraph_d.dst, csrGraph.dst, csrGraph_d.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
    level[srcVertex] = 0;
    cudaMemcpy(level_d, level, csrGraph_d.numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // run kernel
    unsigned int numThreadsPerBlock = 128;
    unsigned int numBlocks = (csrGraph_d.numVertices + numThreadsPerBlock - 1) / numThreadsPerBlock;
    unsigned int newVertexVisited = 1;
    for (unsigned int currLevel = 1; newVertexVisited; ++currLevel)
    {
        newVertexVisited = 0;
        cudaMemcpy(newVertexVisited_d, &newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);
        bfs_kernel<<<numBlocks, numThreadsPerBlock>>>(csrGraph_d, level_d, newVertexVisited_d, currLevel);
        cudaMemcpy(&newVertexVisited, newVertexVisited_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    // copy reuslt to CPU
    cudaMemcpy(level, level_d, csrGraph.numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free mem
    cudaFree(csrGraph_d.srcPtrs);
    cudaFree(csrGraph_d.dst);
    cudaFree(level_d);
    cudaFree(newVertexVisited_d);
}