### Merge
ordered merge - input 2 ordered lists - output 1 ordered list\
find coRanks k in a segment\

very memory bound kernel

### Problems
- due to running binary serches in multiple segments using a single thread in a segment.
- causes a lot of memory divergence
- during corank() function, each thread performs binary search which has random access
- during the sequential merge each thread loops through its own segment of consecutiv elements


### solution to memory divergence here
- if we have uncoalesced accessed to global memory, we can load the data to shared memory in a coalesced way
- then do all the uncoalesced access in the shared memory.
- Load the entire block segment to shared memory.
    - loads fromm global memoory are coalesced.
    - one thread in block does c-rank to find block's input segments
- do the per thread co-rank and merge in shared memory.
    - non coalesced accesses performed in shared memory
- store from shared memory to global memory in a coalesced way.

#### threadcoarsening
thread coarsening already implemented as:
- every thread is assigned a segment of the output.
- instead of assigned 1 element per thread (segment length 1), but its less intuitive.
- by assigning multiple elements to a single thread we amortize the cost of parallelizing ie having to perform binary search over a larger input segment.
    