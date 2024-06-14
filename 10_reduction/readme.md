### Reduction
1. reduction operation reduces a set of input values to one value. eg sum, product ,min, max
- reduction operations is: Assosiative, commutative, have well define identity value

2. problems:
- no. of active threads reduces exponentially every iterations
- threads in the same warp are bound together by SIMD, therefore remain inactive on the SM.
- leads to  control divergence
- memory accesses also are not coalesced ie access varies across threadIdx leading it to not be adjacent when accessed. accesses might not be from the same DRAM Burst, cache evicted? -> memory divergence.

solution:
- coalesced control and memory

3. futher optimization - Shared memory for data reuse:
- make initial load from global memory
- subsequent writes and reads continue in shared memeory
- registers are faster but registers are private to threads. therefore put it in shared memory
- Helps with Data reuse

4. Thread coarsening:
- when hardware is given more thread blocks than it has resources for, it serialises the threadblocks
- if price is paid for parallelization
- we can reduce the cost by rather than letting the hardware serialize the threadblocks, we serialize the threadblocks in the code

Price paid here:
- control divergence
- synchronization

- better to coarsen the threads if there are many more blocks than resources available
- solution: have a threadblock be responsible for an entire segment.

### Disadvangtages of Thread Coarsening:
- we lose the property of transparent scalability
- when we coarsen we interfere with the hardware scheduling
- as we assume hwdr serializes so we serialize it ourselfs in threadcoarsening
- if the hardware doesnt serialize we lose transparent scalability.
- we endup hacving retuning the coarsening factor everytime.

### add boundary conditions