# Setup

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