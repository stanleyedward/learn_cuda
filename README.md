# cudaing
1-7: fundamentals of GPU computing

8-19: parallel patterns

20-23: advanced features

### Setup

1. Setup the CUDA env

```sh
conda create -n cuda
conda activate cuda
conda install -c conda-forge cuda=12.4 gxx
```

2. Run the test code using `nvcc`
```sh
nvcc add.cu -o add_cuda.out
./add_cuda.out
```
`note:` max error should be 0 anything greater try runnning `debug.cu` with `nvprof`

3. Try the profiler
```sh
nvprof ./add_cuda.out
```

### Resources
- [Programming Massively Parallel Processors - 3rd Edition](http://gpu.di.unimi.it/books/PMPP-3rd-Edition.pdf)
- [AUB Spring 2021 El Hajj Programming Massively Parallel Processors](https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4)

- [An Easy Introduction to CUDA C and C++](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
- [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [RTFM](https://docs.nvidia.com/cuda/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [cuda programming - freecodecamp](https://github.com/Infatoshi/cuda-course.git)


#### MISC
- https://chatgpt.com/share/6787c204-ec34-8012-8fd9-790b8258bf6b
