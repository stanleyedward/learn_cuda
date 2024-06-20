### Sort
1. Radix sort (non comparison based)
2. Merge sort (comparison based)

## Radix sort
- non comparison sorting algo
- uses exclusive scan

- extract the bit from the `first digit` of the elemtn using a `bitewise AND` operator to get a `bit array`, which is a array consisting of binary elements [1s and 0s].
- perform `exclusiveScan(bitArray)` to get `# of ones to the left` array
- use the formulas:
    - destination of `zero`: element_idx - #ones to left
    - dest. of `one`: input_size - # ones in total + # ones to the left

- we need to derived this formula in order for it to be depended only on the number of `ones to the left`(we can get `total # of ones` from exclusive sort directly).
- After applying the forumla we get the `destination array` which gives us the destination idx for each element.
- update the locations of each element according to the destination array.
- this will be the input for the next iteration where the bit fro the `next digit` of the elements used

### optimizations in Radix sort
1. memory stores are not coalesced:
- nearby threads write to distant locations in glbal memory

### Solution
- sort locally in shared memory.
- write each buckets of memory in a coalesced manner.

### choice of radix value
- 1-bit radix: was used -> takes 1 bit from each elemtn/key -> 2 buckets
- 2-bit radix: 2-bits from each element -> 4 buckets
    - if our elements/keys are 32-bits
    - 32 iterations to do radix sort
    - N iterations if elements are N-bits long
- to reduce iterations a larger radix can be used.
- Disadvatnge: more buckets->poorer coalescing
- optimization: choice of radix value must be a balance.

another advantage of using 2 or more bits:
- 2bit sort -> is just 2 1-bit sorts
- but since we will be doing 2 sort locally (unlike in 1 bit, where we hav eto copy to global between every 1-bit sort)they end up being more efficient!
- therefore no need to synchronize across other blocks

### issues
1. (UNSOLVABLE problem) sync across blocks needed for the global scan step,that we need after applying the optimization solutions.
- only way to go aroud this is to restart the kernel.
    - 32bit elements with 2 bit radix with take 16 kernel calls. as 16 iterations.
- experience folks can work around it but it ends up very complex.

### thread coarsening
by increasing radix:
- decreasing the ammount of iterations we take
- but makes memory write coalescing worse

- price of parallization across more blocks having smaller buckets per block, hence fewer opportunities to coalesce.
- more blocks, lower # of elements per block, smaller buckets, worse memory coalescing
- if parallized great, but if hardware serializes it better to give each block more work

#### soltuion 
- process more elements per block, larger buckets, more coalescing.
- if `COARSE_FACTOR = 2` then every block is resposible for 2x elements
- we have Block granularity

## Merge Sort

