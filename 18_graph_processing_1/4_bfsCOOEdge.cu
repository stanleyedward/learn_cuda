#include "common.h"
__global__ void bfs_kernel(COOGraph cooGraph, unsigned int *level, unsigned int *newVertexVisited, unsigned int currLevel)
{
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge < cooGraph.numEdges)
    {
        unsigned int vertex = cooGraph.src[edge];
        unsigned int neighbour = cooGraph.dst[edge];
        if (level[vertex] == currLevel - 1 && level[neighbour] == UINT_MAX)
        {
            level[neighbour] == currLevel;
            *newVertexVisited = 1;
        }
    }
}

void bfs_gpu(COOGraph cooGraph, unsigned int srcVertex, unsigned int *level)
{
    // alloc mem
    COOGraph cooGraph_d;
    cooGraph_d.numVertices = cooGraph.numVertices;
    cooGraph_d.numEdges = cooGraph.numEdges;
    cudaMalloc((void **)&cooGraph_d.src, cooGraph_d.numEdges * sizeof(unsigned int));
    cudaMalloc((void **)&cooGraph_d.dst, cooGraph_d.numEdges * sizeof(unsigned int));

    unsigned int *level_d;
    cudaMalloc((void **)&level_d, cooGraph_d.numVertices * sizeof(unsigned int));
    unsigned int *newVertexVisited_d;
    cudaMalloc((void **)&newVertexVisited_d, sizeof(unsigned int));
    cudaDeviceSynchronize();

    // copy data to GPU
    cudaMemcpy(cooGraph_d.src, cooGraph.src, cooGraph_d.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cooGraph_d.dst, cooGraph.dst, cooGraph_d.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
    level[srcVertex] = 0;
    cudaMemcpy(level_d, level, cooGraph_d.numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // run kernel
    unsigned int numThreadsPerBlock = 256;
    unsigned int numBlocks = (cooGraph_d.numEdges + numThreadsPerBlock - 1) / numThreadsPerBlock;
    unsigned int newVertexVisited = 1;
    for (unsigned int currLevel = 1; newVertexVisited; ++currLevel)
    {
        newVertexVisited = 0;
        cudaMemcpy(newVertexVisited_d, &newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);
        bfs_kernel<<<numBlocks, numThreadsPerBlock>>>(cooGraph_d, level_d, newVertexVisited_d, currLevel);
        cudaMemcpy(&newVertexVisited, newVertexVisited_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    // copy reuslt to CPU
    cudaMemcpy(level, level_d, cooGraph.numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free mem
    cudaFree(cooGraph_d.src);
    cudaFree(cooGraph_d.dst);
    cudaFree(level_d);
    cudaFree(newVertexVisited_d);
}