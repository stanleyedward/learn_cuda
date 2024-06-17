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

### Solution in GPU
- Atomic OPerations: opers on GPU that perform read-modify-write with a single ISA instruction(intrinsics).
- ie the hardware gaurantees that no other thread can access the memory location until theoperation completes
- ie concurrent atomic operations to the same memory lcoation are serialized by the hardware itself. not the software

Types of atomic operations in CUDA.
- atomic add: T atomicAdd(T* address, T valToAdd)
- sub, min, max, inc, dec, and, or , xor, exchnage, commpare, swap.