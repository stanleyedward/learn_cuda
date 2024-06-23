#include "common.h"
#define LOCAL_QUEUE_SIZE 2048 // 256 threads per block, 2048 threads in total or 8 blocks, and we have 96KB in a v100 GPU, therefore we want each thread block not to use more than 12kb of shard memory.

__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int *level, unsigned int *prevFrontier, unsigned int *currFrontier, unsigned int numPrevFrontier, unsigned int *numCurrFrontier, unsigned int currLevel)
{
    __shared__ unsigned int currFrontier_s[LOCAL_QUEUE_SIZE];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0)
    {
        numCurrFrontier_s = 0;
    }
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier)
    {
        unsigned int vertex = prevFrontier[i];
        // loop over the outgoing edges
        for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge)
        {
            unsigned int neighbour = csrGraph.dst[edge];
            if (atomicCAS(&level[neighbour], UINT_MAX, currLevel) == UINT_MAX) // atomicCAS returns the old value
            {
                unsigned int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1);

                // we may reach max limit of idx in the currFrontier_s as we only have 2048 statically alloted
                if (currFrontierIdx_s < LOCAL_QUEUE_SIZE)
                {
                    currFrontier_s[currFrontierIdx_s] = neighbour;
                }
                else
                {
                    numCurrFrontier_s = LOCAL_QUEUE_SIZE;
                    unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                    currFrontier[currFrontierIdx] = neighbour;
                }
            }
        }
    }
    __syncthreads();
    // commit private queue to the global queue
    __shared__ int currFrontierStartIdx;
    // we only want 1 thread to do this
    if (threadIdx.x == 0)
    {
        currFrontierStartIdx = atomicAdd(numCurrFrontier, numCurrFrontier_s);
    }
    __syncthreads();
    for (unsigned int currFrontierIdx_s = threadIdx.x; currFrontierIdx_s < numCurrFrontier_s; currFrontierIdx_s += blockDim.x)
    {
        currFrontier[currFrontierStartIdx + threadIdx.x] = currFrontier_s[currFrontierIdx_s];
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