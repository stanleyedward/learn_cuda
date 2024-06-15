## Scan brent-kung method
- to address low work efficiency of koggleStone scan

- kogge-stone
a. Log(N) steps b. O(N*logN) operations

- brent-kung
a. reduction step
- LogN steps
- N-1 operations

b. post reduction step
- logN - 1 steps
- N-2 - (LogN - 1)

c. Total
- 2*log(N) - 1steps
- O(n) operations

- takes more steps but is more work efficient


## optmizations similar to kogge-stone
1. shared memory: data reuse, memory coalescings
2. doubleBuffering : is not needed

problem:
- memory was coalesced in kogge-stone, therefore using shared memory is even more effective
- doubleBUffering is not needed as dont write and read the same element in the same iterations therefore single iteration is okay
- control divergence is a problem in brentKung unlike koggeStone

## Solution to minimize control divergence
- do not assign threads to specific data elements
- re-index threads on every tieration to differnet elements(general approach to reduce control divergence)



## Exclusive Scan
2 ways to do exclusive scan
1. formulaate as inclusive scasn just like kogge stone by shifting the leemnts 1 input to the left.
2. using a different post-reduction step.


## work efficiency (the reality)
- even tho brent kung has more theoretical workefficient 
- it ends up being slower than  kogge stone while running in the GPU.
- the work being saved in the brent-kung is being done by threads bound by SIMD anyways. 
- when im using multiple warps we are saving work but
- inside a single warp using fewer threads in the warp doesnt benefit us. as the inactive threads still occupy resources due to the way SIMD works.
- in practice the after accounting for the inactive threads it ends up being O(N* logN)
- performance may be equal or even worse than kogge stone approach.

## Thread Coarsening
- the optimization that is useful when there si aprice to parallelize something
- other than control divergence, synchronization
- Parallelzing scan incurs the overhead of lowering work efficiency
- our work efficiency is low
- if the resources are insufficent and the hardware serializes the threadblocks, it may incur ovvrehead unncescessarily.
- therefore we serialize it ourselves by applying thread coarsening

## Solution to low work efficiency is Thread coarsening
- in the context of scan, is via segmented scan
- ie each thread scans a segment sequentially (which is more work efficeint)
-  only scan partial sums of each segment in parallel