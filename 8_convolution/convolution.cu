#include "common.h"
#define OUT_TILE_DIM 32


__constant__ float mask_c[MASK_DIM][MASK_DIM]; //constant = cannot write to in in the GPU, but can copy it to GPU from the CPU.
__global__ void convolution_kernel(float* input, float* output, unsigned int width, unsigned int height){
    int outputRow = blockDim.y*blockIdx.y + threadIdx.y;
    int outputCol = blockDim.x*blockIdx.x + threadIdx.x;

    //boundry conditions
    if(outputRow < height && outputCol < width){
        float sum = 0.0f; 
        for(int maskRow = 0; maskRow < MASK_DIM; ++maskRow){
            for(int maskCol = 0; maskCol < MASK_DIM; ++maskCol){
                int inputRow = outputRow - MASK_RADIUS + maskRow;
                int inputCol = outputCol - MASK_RADIUS + maskCol;
                if((inputRow < height && inputRow >= 0) && (inputCol < width && inputCol >=  0)){
                    sum += mask_c[maskRow][maskCol]*input[inputRow*width + inputCol];
                }
            }
        }
        output[outputRow*width + outputCol] = sum;
    }

}

void convolution_gpu(float mask[][MASK_DIM], float* input, float* output, unsigned int width, unsigned int height){
    //alloc mem
    float *input_d, *output_d;
    cudaMalloc((void**) &input_d, width*height*sizeof(float));
    cudaMalloc((void**) &output_d, width*height*sizeof(float));
    cudaDeviceSynchronize();

    //copy data to gpu
    cudaMemcpy(input_d, input, width*height*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_d, output, width*height*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    //copy mask to gpu
    //we can only allocate upto 64KB, input is also constant but it is too large to put in constant memory
    cudaMemcpyToSymbol(mask_c, mask, MASK_DIM*MASK_DIM*sizeof(float));
    cudaDeviceSynchronize();

    //run kernel
    dim3 numThreadsPerBlock(OUT_TILE_DIM, OUT_TILE_DIM, 1);
    dim3 numBlocks((width + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (height + OUT_TILE_DIM - 1)/OUT_TILE_DIM);
    convolution_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, width, height);
    cudaDeviceSynchronize();

    //copy mem to cpu
    cudaMemcpy(output, output_d, width*height*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //free mem
    cudaFree(input_d);
    cudaFree(output_d);
    cudaDeviceSynchronize();
}