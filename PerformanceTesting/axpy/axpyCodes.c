//******************************************************************************** */
// F -> COpenMP

1.
Code Block 7:
void saxpy_parallel(int n, float a, float *x, float *y) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

2.
Code Block 7:
void saxpy_parallel(int n, float a, float *x, float *y) {
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

3.
Code Block 7:
void saxpy_parallel(int n, float a, float *x, float *y) {
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}
//********************************************************************************
// F -> C(OpenACC)

4.
Code Block 6:
void saxpy_parallel(int n, float a, float *x, float *y) {
    #pragma acc kernels loop present(x, y)
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

5.
Code Block 1:
void saxpy_parallel(int n, float a, float *x, float *y) {
    #pragma acc parallel loop present(x[0:n], y[0:n])
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

6.
Code Block 2:
void saxpy_parallel(int n, float a, const float *x, float *y) {
    int i;
    #pragma acc parallel loop present(x, y)
    for (i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}
//********************************************************************************
// FOpenMP -> C HIP
7.
Code Block 2:
void saxpy_parallel(int n, float a, float* x, float* y) {
    float* d_x;
    float* d_y;

    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));

    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(saxpy_kernel, dim3(grid_size), dim3(block_size), 0, 0, n, a, d_x, d_y);

    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_x);
    hipFree(d_y);
}

__global__ void saxpy_kernel(int n, float a, float* x, float* y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

8.
Code Block 2:
void saxpy_parallel(int n, float a, const float *x, float *y) {
    float *d_x, *d_y;

    // Allocate device memory
    hipMalloc((void**)&d_x, n * sizeof(float));
    hipMalloc((void**)&d_y, n * sizeof(float));

    // Copy data to device
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch the SAXPY kernel
    hipLaunchKernelGGL(saxpy_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, n, a, d_x, d_y);

    // Copy result back to host
    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_x);
    hipFree(d_y);
}

__global__ void saxpy_kernel(int n, float a, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

9.
Code Block _:
__global__ void saxpy_parallel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
//********************************************************************************
// F(OpenMP) -> C CUDA
10
Code Block 3:
void saxpy(int n, float a, float *x, float *y) {
    float *d_x, *d_y;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    saxpy_parallel<<<gridSize, blockSize>>>(n, a, d_x, d_y);

    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
}

__global__ void saxpy_parallel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

11.
Code Block 2:
void saxpy_parallel(int n, float a, float *x, float *y) {
    // Allocate device memory for x and y
    float *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    // Copy vectors x and y to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Configure launch parameters
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch kernel on the device
    saxpy<<<numBlocks, blockSize>>>(n, a, d_x, d_y);

    // Copy result back to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

12.
Code Block _:
__global__ void saxpy_parallel(int n, float a, float *x, float *y) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = a * x[idx] + y[idx];
  }
}
//********************************************************************************


