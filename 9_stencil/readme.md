Stencil is a special case of convolution, thats why its very simimlar to it.
- for ease of understanding we wont compute the edge elements to avoid having to code in boundary conditions

- we can optimize further using tiling to reduce redundant accesses by loading it into shared memory, as threadsin the same block load some of the same input elements 
- therefore have each threads lead one input element to shared memory and other threads access the element from shared memory

problem: similar to tiliing in convolution
- input and output tile have different dimensions
- output tiles  dimensions = input tile dim - 2
- launch enough threads per block to load the input tile to shared memory.
- th en only use a subset ofthem to compute and dstore the output tile.

- tiling ended up slightly slower?\
-lets analyze why\
1. in the originalkernel:
- each thread performed 8 operations 2 multiplications and 6 additions (8 OP 2FP muls and 6FP adds)
- each thread loaded 7 different values each 4bytes = 28bytes (7FP values 28B)
- thereforce compute-memory ratio = 0.29 OPs/B

2. inthe tiled kernel:
- input tile size is T
- output tile is size T-2
- each block performs (8 OPs)*(T-2)^3 OPs
- each block loads (4B)*T^3
- Ratio: 2*(1-2/T)^3\
theerfor for T=8 the ratio is 0.84 OPs/B, the improvement is not that great as the overhead is high\
- but if we increase T we improve this ratio eg. if T=32 the ratio is 1.65(2x improvement).
- as boundry elements have lower data reuse, and increase the tile size decreases the ratio of boundary elemetns to total elements
- but we cant increase the block dim above 8 as we set it\
1. input tile size same as block size which is limite dby hardware
2. need mmore shared memory to store the input.
3. too much shared memory used may hurt occupancy.

- solution: thread coarsening
- to process a larger input/output tile without using more threads
- thread coarsening used when there is aprice of parallelization , here it is redundant loading of boundary elements.
- Tile Slicing: keep only needed slices of the input tile in shared memory.
