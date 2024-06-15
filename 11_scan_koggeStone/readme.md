## Scan - Kogge Stone method
## optimization: double-buffering

takes: 1. input array 2. associative operator 
returns: 1. output array where
- inclusive scan - preceeding and corresponding element
- exclusive scan - only preceeding elements

1. Segmented Scan
2. Shared memory
- as memory locations are reused
- use shared memory
3. Double buffering
- every loop iterations we needed syncthreads twice to make use everybody reads before everybody writes.
- as we are using the same buffer for input and output
SOlution: 
- doubleBuffering: have 2 different buffers. alternating them for input and output so we only have 1 sync()
- inactive threads copy the value to the next buffer

4. Exclusive kogge stone scan

