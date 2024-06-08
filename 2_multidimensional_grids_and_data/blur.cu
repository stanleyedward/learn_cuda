  __global__ void blur_kernel(unsigned char *image, unsigned char* blurred, unsigned int width, unsigned int height){
   // assign one thread to each output px and and read multiple input px
   int outputRow = blockIdx.y * blockDim.y + threadIdx.y; //cannot be unsigned int as it results in weird behaviour in the subtraction
   int outputCol = blockIdx.x * blockDim.x + threadIdx.x;
   int BLUR_SIZE = 3;
   if (outputRow < height && outputCol < width){
      unsigned int average = 0;
      for (int inputRow = outputRow - BLUR_SIZE; inputRow < outputRow + BLUR_SIZE + 1; ++inputRow){
         for(int inputCol = outputCol - BLUR_SIZE; inputCol < inputRow + BLUR_SIZE +1; ++inputCol){
            // Boundry condition
            if(inputRow < height && inputRow >=0 && inputCol < width && inputCol >= 0){
            average += image[inputRow*width + inputCol];
            }
         }
      }
      // normally we make px on the boundries darker cause the boundry condition we 
      //may not have considered the entire kernel for the average in edges and corners
      blurred[outputRow*width + outputCol] = (unsigned char) average/(2*BLUR_SIZE + 1)*(2*BLUR_SIZE+1);
   }

 }

 void blur_gpu(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height){
    // Allocate GPU mem
    unsigned char *image_d, *blurred_d;
    cudaMalloc((void**) &image_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &blurred_d, width*height*sizeof(unsigned char));
    
    cudaDeviceSynchronize();

    // Copy data to GPU
    cudaMemcpy(image_d, image, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // call the blur kernel
    dim3 numThreadsPerBlock(16,16, 1);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1)/ numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1)/ numThreadsPerBlock.y, 1);
    blur_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, blurred_d, width, height);
    cudaDeviceSynchronize();

    // copy data from GPU to CPU
    cudaMemcpy(blurred, blurred_d, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free Memory
    cudaFree(image_d);
    cudaFree(blurred_d);
 }