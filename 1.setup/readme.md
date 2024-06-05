# Setup

1. Setup the CUDA env

```sh
conda create -n cuda
conda activate cuda
conda install -c conda-force gxx cuda
```

2. Run the test code using `nvcc`
```sh
nvcc add.cu -o add_cuda.out
./add_cuda.out
```

3. Try the profiler
```sh
nvprof ./add_cuda.out
```