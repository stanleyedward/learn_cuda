#ifndef COMMON_H
#define COMMON_H
struct CSRGraph
{
    unsigned int numVertices;
    unsigned int numEdges;
    unsigned int* srcPtrs;
    unsigned int* dst;
};

void bfs_gpu(CSRGraph csrGraph, unsigned int srcVertex, unsigned int* level);

#endif //COMMON_H