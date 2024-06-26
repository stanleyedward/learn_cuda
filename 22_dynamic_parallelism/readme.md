## Dynamic Parallelism
- in CUDA, refers to the ability of threads executing on the GPU to launch new grids (nested parallelism)
- DP is useful for nested paralleism
- useful when the amount of nestedd work is unknown, so cannot be launched up front.
    - nested parallel work is irreular
        - grph algorithms (each vertex has a diff no of neibours)
        - bezier curves (each curve needs different # points to draw)
    - nested parallel work is recursive with unknown depth.
        - tree traveral algorithms (quadtrees and octrees)
        - divide and conquer algo (quicksort)
    
### example in Frontier BFS
- we can parallize this loop: 
```c
        unsigned int vertex = prevFrontier[i];
        unsigned int start = csrGraph.srcPtrs[vertex];
        unsigned int numNeighbours = csrGraph.srcPtrs[vertex + 1] - start;
        // loop over the outgoing edges
        for (unsigned int i = 0; i < numNeighbours; ++i)
        {
            unsigned int edge = start + i;
            unsigned int neighbour = csrGraph.dst[edge];
            // using atomic operations to prevent raiseconditions for reasons above
            if (atomicCAS(&level[neighbour], UINT_MAX, currLevel) == UINT_MAX) // atomicCAS returns the old value
            {
                unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                currFrontier[currFrontierIdx] = neighbour;
            }
        }
```
device code for calling a kernel to launch a grid is same as the host code

- very large amount of launches may not be able to run at the same time, therefore their state must be saved to a buffer before it runs, therefore it needs memory

- limit for no of dynamic launches is `pending launch count`
- default count is 2048. exceeding causes errors.
- limit can be changed with
```c
cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, <new limit>);
```


### results
performance was actually worse 
- why? the grids might have new low amount of neighbours therefore these grids are relatively small, + we have a large amount of grids

### streams
- related to streams
on the CPU we had stream, therefore grids in parallel must be on different streams
    -  wihotu specifynig a stream whencallin a kernel, grids get launched into a default stream.
    - on gpu, threads in the same block sahre the same default stream
    - laches by threads ono the same blocka re serialized.

Parallelism can be improved by createing a differnt stream per thread.
- 2 approaches to do it
    - use the stream API like before in 21_pinMem_streams
    - compiler flag `--default-stream per-thread`

results: performance only improves by little, as here theres other problems

### optimization
- very small grids may not be worth the overhead(more efficient to serialize).
- smaller grids -> thread blocks maybe underutilize the SM.
- queueing delays.
 
solution: 
- apply a threshold: 
    - only launch large grids that are worth the overhead
    - threshold value is data dependent.
- aggregates launches (complicated):
    - one thread collect the work of multiple threads and launch a single grid on their behalf.

### results perform much betters
