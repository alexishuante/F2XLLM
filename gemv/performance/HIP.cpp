#include <hip/hip_runtime.h>
#include <stdio.h>

// Declare global variables to measure time
hipEvent_t start, stop;
float totalMilliseconds = 0;


// HIP kernel for generalized matrix-vector multiplication (GEMV)
__global__ void gemvKernel(int n, const float* A, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

// Host function for GEMV using HIP
void gemvParallelHIP(int n, const float* A, const float* x, float* y) {
    // Device pointers
    float *d_A, *d_x, *d_y;

    // Allocate memory on the GPU
    hipMalloc((void**)&d_A, n * n * sizeof(float));
    hipMalloc((void**)&d_x, n * sizeof(float));
    hipMalloc((void**)&d_y, n * sizeof(float));

    // Copy data to the GPU
    hipMemcpy(d_A, A, n * n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);

    hipEventCreate(&start);
    hipEventCreate(&stop);



    

    // Launch the kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    hipEventRecord(start);
    hipLaunchKernelGGL(gemvKernel, dim3(gridSize), dim3(blockSize), 0, 0, n, d_A, d_x, d_y);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    // Calculate the time it took to run the kernel
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    totalMilliseconds += milliseconds;

    // Print the time it took to run the kernel
    printf("Time: %.3f ms\n", milliseconds);

    printf("hi\n");



    // Copy the result from the GPU
    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);

    // Free GPU memory
    hipFree(d_A);
    hipFree(d_x);
    hipFree(d_y);
}

int main() {
    // Warmup
    int n = 43346;
    float* A = new float[n * n];
    float* x = new float[n];
    float* y = new float[n];

    // Initialize A and x with some values
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0f;
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = (i + j) % 100;
        }
    }

    // Warm-up run
    gemvParallelHIP(n, A, x, y);

            // Generate random input data
    for (int j = 0; j < n * n; ++j) {
        A[j] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int j = 0; j < n; ++j) {
        x[j] = static_cast<float>(rand()) / RAND_MAX;
    }



    // Measure time for 10 runs
    int numRuns = 10;
    totalMilliseconds = 0;
    for (int i = 0; i < numRuns; ++i) {
        // Call the function and measure time
        gemvParallelHIP(n, A, x, y);
    }

    // Print the average time
    printf("Average time over %d runs: %.3f ms\n", numRuns, totalMilliseconds / numRuns);
    

    // Clean up
    delete[] A;
    delete[] x;
    delete[] y;

    return 0;
}