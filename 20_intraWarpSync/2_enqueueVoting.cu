#define WARP_SIZE 32
#include "common.h"
__global__ void enqueue_kernel(unsigned int *input, unsigned int *queue, unsigned int N, unsigned int *queueSize)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        unsigned int val = input[i];
        if (cond(val))
        {
            // contention due to atomic operations
            unsigned int j = atomicAdd(queueSize, 1);
            queue[j] = val;
        }
    }
}

__global__ void enqueue_voting_kernel(unsigned int *input, unsigned int *queue, unsigned int N, unsigned int *queueSize)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        unsigned int val = input[i];
        if (cond(val))
        {
            // assign leader thread by identifying which threadss are active
            unsigned int activeThreads = __activemask();    // return 32 bit integer that has a bit set for each active thread
            unsigned int leader = __ffs(activeThreads) - 1; // find first set returns first active bit in the mask
            // how many threads need to add?
            unsigned int numActive = __popc(activeThreads); // returns no. of active bits in the mask, here gives us no of active threads
            // leader allocate in queue
            unsigned int j;
            if (threadIdx.x % WARP_SIZE == leader)
            {
                j = atomicAdd(queueSize, numActive);
            }
            // broadcast j's value to other threads at the warp
            j = __shfl_sync(activeThreads, j, leader);

            // find the offset of each active thread in the queue
            unsigned int previousThreads = (1 << (threadIdx.x % WARP_SIZE)) - 1; // mask for prev threads
            unsigned int previousActiveThreads = activeThreads & previousThreads;
            unsigned int offset = __popc(previousActiveThreads);

            // store the result in the queue
            queue[j + offset] = val;
        }
    }
}