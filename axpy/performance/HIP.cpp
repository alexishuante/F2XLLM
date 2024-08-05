#include <stdio.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

//Declare global variables to measure time
hipEvent_t start, stop;
float totalMilliseconds = 0;


__global__ void saxpy_kernel(int n, float a, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

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
    // Use global variables start and stop to measure time
    hipEventRecord(start);
    hipLaunchKernelGGL(saxpy_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, n, a, d_x, d_y);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    // Make the global variable totalMilliseconds the time it took to run the kernel
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    totalMilliseconds += milliseconds;


    




    // Copy result back to host
    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_x);
    hipFree(d_y);
}


int main() {
    int n = 925000000; // Size of vectors
    float a = 2.0f; // Scalar value for saxpy
    float *x, *y;

    // Allocate memory for x and y on the host
    x = (float*)malloc(n * sizeof(float));
    y = (float*)malloc(n * sizeof(float));

    // Initialize x and y with some values
    for(int i = 0; i < n; i++) {
        x[i] = 1.0f; // Example value
        y[i] = 2.0f; // Example value
    }

    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Warmup run
    saxpy_parallel(n, a, x, y);

    // 10 consecutive runs
    for(int i = 0; i < 10; i++) {
        saxpy_parallel(n, a, x, y);
    }

    // Calculate average time
    totalMilliseconds /= 10;
    printf("Average time: %f ms\n", totalMilliseconds);
    

    // Cleanup
    free(x);
    free(y);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    return 0;
}
