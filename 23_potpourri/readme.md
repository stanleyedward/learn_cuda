## Potpourri
### MultiGPU programming
- multiple GPU on the same node. - multiple CPU threads each driving one GPU (typically OpenMP)
- multiple GPUS on multiple nodes - MPI(message passing interface) 

### Interconnect
- PCIe usedin many systemsto connect CPUs and GPUs together
- NVLink provides faster interconnect between multiple GPUs on the same system

### Memory Management
- Unified virtual Addressing(UVA)
    - 2 idfferent physical memory spaces
    - but unified virtual address spaces across host and device.
    - data location is known from the pointer value, itself.
    - no need to specify copies directions ie from host to device, vice versa.
    - `cudaMemcpyDefault()`

- Zero-copy memory
    - enables ddevice thresds to directly access host memory.
    - no cudamemcpys needed
    - runtime system copies it on demand
    - uses DMA to copy the data.    
    - allocated differnetly than normal
    - data that has to accessed from the GPU thread in the CPU memory must be pinned `cudaHostAlloc(&ptr, size, cudaHostAllocMapped)`
    - `cudaHostgetDevicePointer()` to get corrspodning device pointer
        - unnecesarry if system supports UVA

    - kind of provides automatic pipelining as threads ccess memory on demand.
    - but with cudamemcpy: we copy in bulk, so some overhead might be amortized by the systems when its in bulk.
    - depends on situation which is better.

These are  used bery widely.
### Events
- we have been synchronizing fter every copy or kernel call to collect time computation.
    - in practise this apprach interferees with executino and slows things down.
- Event can b usedd to collect timing information without synchronizing.
    - `cudaEventCreate(&event)`
    - `cudaEventRecord(event, stream=0)`
    - `cudaEventSynchronize(event)`
    - `cudaEventElapsedTime(&result, startEvent, endEvent)`
    - `cudaEventDestroy(event)`
Profiler can also be used.

### Tensor cores
- tensor cores are programmagble matmul andaccumulate units that run in a single instruction inside an SM. introduced with the Volta architecture
    - usefil for DNN workloads
In volta V100:
    - 640 tnesor cores(8 per SM)
    - each core is capable of a 4x4 matmul: D = A*B + C

### Libraries
many libraries have common parallel primitives/patterns.
    - thrust: reduction, scan, filter, and sort
    - cuBLAS: basic dense linear algebra subprograms
        - nvBLAS: mutliGPU on top of cuBLAS
    - cuSPARSE: sparse-dense linear algebra
    - cuSOLVER: factorization and solve routines
        - top of cuBLAS and cuSPARSE
    - cuDNN: deep neural networks
        - used by caffe, MxNet, tensorflow, pytorch.
    - nvGRAPH: graph processing
    - cuFFT: fast fourier transofmr
    - NPP: signal, imamge and video processing.

### other programming interfaces
we focused on CUDA
- oterh interfaces
    -  OpenCL: open-source, support nvidia and non_nvidia GPUs
        - support differnt GPUs and hardware.
    - OpenACC: directive-based GPU programming
        - annotate code similar to OpenMP(also works on cpu)
    - C++AMP: library for implementing GPU programs directly in C++.

### Other hardware.
we focused on discrete(seperate from CPU chip) NVIDIA GPUs.
- AMD (Radeon)
- ARM (Mali)
- Intel

Integrated GPUs
- CPU and GPU on the same chip(same physical memory), AMD is a pioneer for integrated GPUs
- no memory copes or data transfer over PCIe.

### Comparision with CPU
disclaimer:
comparison of GPU (runs warps in parallel), with CPU using single threaded nonVectorized code. has been done
    - exaggerates GPU speedup
    - frowned upon generally in research papers.

For fair to compare with parallel and vectorized CPU code. (threads run in parallel)