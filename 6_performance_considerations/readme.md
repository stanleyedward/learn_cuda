Performance considerations
1. DRAM Banks\
2. Memory coalesing\
3. Thread Granularity\
4. optimized tiling MatMul - 
    - in this one thread block is responsible for processing multiple output tiles
    - sequentially and reuse the input tile that it loaded
5. Thread Coarsening - a singlel thread is assigned multiple units of parallism