### Convolution
- convolutional filter/kernel/mask
- constant memory - upto 64kb - more efficient cache - read only as its constant
- no need to manage dirty bits, write backs, tracking changes, etc.
- no need for cache coherence which is used in writeable cache.
- small size: minimize evictions => low miss rate
- constant cache is managed by the hardware NOT by the programmer unlike shared memory
- we can use tiling to further speedup
- to do boundry conditions in tiling we can either
    1. check if access is inbounds
    2. or use ghost elements ie assign to 0
    3. padding