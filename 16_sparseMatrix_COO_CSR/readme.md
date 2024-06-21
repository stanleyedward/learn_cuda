## Sparse Matrix Computation
case study: sparse matrix-vector mul (SpMV)\
Storage Formats:
- COO: coordinate format
- CSR: compressed sparsed row
- ELL: ELLPACK format
- JDS: Jagged Diagonal Storage
 
sparse matrix: majority elements are zero
(many real world systems are sparse, almost < 1% are non-zerof).
- no space allocated for zeros(save memory)
- no need to load zeros(save memory bandwidth)
- no need to compute with zeros(save compute time)

format design considerations:
- space efficiency(memory consumed)
- flexibility (ease of adding/ reordering elements)
- Accessibility (ease of finding desired data)
- memory access pattern (enable memory coalescing)
- load balance (minimize control divergence)

Choice of best format depends on the computation
- use SpMV computation as an example to study differnt formats
    - [1 Sparse Matrix]*[1 Dense Vector] = [1 Dense Vector]

### COO: coordinate format
![alt text](coo.png)
