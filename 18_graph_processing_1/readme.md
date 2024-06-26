## Graph Processing 1

### Representing Graphs
graphs represented as adjecency matrices.\
    - we can use same storage format as sparse matrices.

we are working with unwieghted and undirected graphs
    - non zeros are all ones
    - matrices are symmetric (CSR/CSC are equivalent)

![alt text](coo_csr_representatoin.png)

### approaches to parallelize
- vertex-centric: 1 thread per vertex
    - typucally CSR/CSC, as given row easy to find non zero values, here given vertex easy to fina l its neighbours.
    - can use ELL and JDS as optimizations

- edge centric: 1 thread per edge
    - typically COO, as given non zero easy to find row and col idxs
    - given an dege, easy to find its source and destination vertices.

 - Hybrid:
    - ex: given a edge, find neigbours of the source vertices and neighbuors of the destination vertices.
    - uses both COO and CSR
    - application: Triangle counting, k-clique decomposition
 
### BFS breadth-first-search
- Vertix centeric has 3 approaches
    - top-down: assign thread to every parent vertex in teh BFS Tree - kernel time is quite slow
    - bottom-up: assign thread to every potential child vertex in the BFS tree. - *faster than top-down as loops can break early if the vertex finds a neighbour.
    - direction optimized - Hybrid approach: starts topdown then switches the the bottom up approach
        - as bottomm up is inefficient in the beginning most vertices will search all neighbours.
        - at the beginnning the top down will just loop over the neighbouts of the first source vertex. being *less efficient to find the first level vertices.

- Edge Centric:
    - 1 thread per edge, check if source vertex of the edge was in the prev lvl, then add the destination vertx to the current lvl.
    - has no for loops
    - better than top down and bottom up but not than direction optimize in our case*.

### dataset implications
> - *our example were testing on a high degree graph(eg social network graph).
- best parallization approach depends on the structure of the graph.
- vertex centric bottomup and edge centric appraches are better on high degree graphs ex: social network graphs.
    - as they are better at dealing with load imbalance
- vertex centric topdown approach is actually better on low degree graphs eg. geographic map graph.
    - vertex centric topdown performs much better here, as in low degree graphs -> # of vertices we visit each iteration is small as we have a low degree.

    - further optimziations: launch more threads to prcoess neighbors of high-degree vertices.(later)


### similarities between BFS and SpMV

 ![alt text](similarities.png)

 - linear algebric forumlation of graph problems
 - witha few tweaks BFS can be formulated EXACTLY as SpMV
 - infact many graph problems canbe formulated din terms of sparse linear algebra computations

- Advantage:
    - u can leverage mature high performance well-optimized libraries for linear algebra.
    - linear algebra operations are easy to parallize
- Disadvantage
    - not always the most efficient way to solve a graph problem.

### Redundant work
- in all our appraches to far, we have redundant work.
strengths: easy to implement, highly paralle, no synchronization across threads.\
weakness: redundant work, many threads will find that their vertex or edge is not relevant for this iteration and just exit.




    

