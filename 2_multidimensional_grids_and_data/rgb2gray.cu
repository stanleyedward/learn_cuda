__global__ void rgb2gray_kernel(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray, unsigned int width, unsigned int height){
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // since R G and B in C memory is stored in 1d and using row major order, to get 1D index from 2D
    unsigned int index = row*width + col;
    if (row < height && col < width){
    gray[index] = red[index]*3/10 + green[index]*6/10 + blue[index]*1/10; 
    }
}


void rgb2gray_gpu(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray, unsigned int width, unsigned int height){
    // Allocate GPU memory
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    cudaMalloc((void**) &red_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &green_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &blue_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &gray_d, width*height*sizeof(unsigned char));

    cudaDeviceSynchronize();

    //cpoy data to GPU
    cudaMemcpy(red_d, red, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();


    // Call kernel
    dim3 numThreadsPerBlock(32, 32, 1);// an int dtype with 3 dim x, y and z
    dim3 numBlocks((width + numThreadsPerBlock.x -1)/numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y, 1);
    rgb2gray_kernel <<<numBlocks, numThreadsPerBlock>>> (red_d, green_d, blue_d, gray_d, width, height);

    cudaDeviceSynchronize();

    // copy data from the GPU to CPU
    cudaMemcpy(gray, gray_d, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); 

    // free GPU memory
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
    cudaDeviceSynchronize();
}

