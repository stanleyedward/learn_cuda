## Histogram
new feature: atomic operations
new optimization: privatization

- histogram approximates the distribution of a dataset\
ex: color histogram

1. assign every threads to 1 input px\
Data Races: when different threads accessing the same output value in the same memory location concurrently without ordering and atleast 1 of these accesses is a write, data races start to happen; as ++bins[b] is not a single operation that happens it consists of multiple instructions(not atomic).
- data races can lead to unpredictable program output

++bins[b] => oldVal = bins[b]; newVal = oldVal + 1; bins[b] = newVal;

### Solution to data races:
- Mutual Exclusion: concurrent read-modify-write opers to the same mem location need to be made matually exclusive to enforce ordering
ex: CPU have mutex locks

```c
mutex_lock(lock);
++bins[b];
mutex_unlock(lock);
```

- therefore each bin value has its own mutex lock (on CPU)
`note:` this is a very bad idea on GPU. why?

### locks and SIMD execution
assume thread 0 and thread 1 in the SAME warp try to acquire the same lock
- thread 0 -> acquired lock 
- thread 1 -> waiting for thread 0 to release lock
- thread 0 -> waits for thread 1 to complete previous instruction (SIMD model)
as all threads in a warp follow SIMD
- we add a lock to each bin.

### Solution in GPU
- Atomic OPerations: opers on GPU that perform read-modify-write with a single ISA instruction(intrinsics).
- ie the hardware gaurantees that no other thread can access the memory location until theoperation completes
- ie concurrent atomic operations to the same memory lcoation are serialized by the hardware itself. not the software

Types of atomic operations in CUDA.
- atomic add: T atomicAdd(T* address, T valToAdd)
- sub, min, max, inc, dec, and, or , xor, exchnage, commpare, swap.

### optmization: privatization
- atomic operations on global memory have high latency
- as we need to wait for both read and write to complete not only modify
- wait if there are othre threads accessing the same location

- privatization is commonly used when different threads contend on a shared output.
- here create a privation copy of the histogram for each thread block
- each thread block will update its private copy of the histogram.
- after the threadblock is done executing
- they can commit its private copy histogram to the global copy atomically.

- privatization only works for assosiative and commutative operations

- Advantages of privatization:\
1. reduces contention in global copy
2. if output is small enoguh private copy can be in shared memory.

### Thread Coarsening
- the price paid of parallelization here is: having more and more threadblocks in parallel, we have more and more private copies needed to be commmited to global copy.
- if these thread blocks are parallel its worth it, but if its gonna be serialized its worth to have fewer thread blocks, and fewer private copies.
