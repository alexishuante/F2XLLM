#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>


// Define global variables to measure time
cudaEvent_t start, stop;
float total_time = 0;


// CUDA Kernel for GEMV operation
__global__ void gemv_kernel(int n, const float *A, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

// Host function to call the GEMV kernel
void gemv_parallel(int n, const float *h_A, const float *h_x, float *h_y) {
    float *d_A, *d_x, *d_y;

    // Allocate device memory
    cudaMalloc((void**)&d_A, n * n * sizeof(float));
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define thread block size and grid size
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Use global variables to measure time and make sure it is synchronized with CUDADeviceSynchronize
    cudaEventRecord(start); // Record start event
    gemv_kernel<<<gridSize, blockSize>>>(n, d_A, d_x, d_y);
    cudaEventRecord(stop); // Record stop event

    cudaEventSynchronize(stop);
     // Wait for the kernel to finish
    float millisecondsTemp = 0;
    cudaEventElapsedTime(&millisecondsTemp, start, stop); // Calculate elapsed time
    printf("Time taken for kernel execution: %f ms\n", millisecondsTemp);
    total_time += millisecondsTemp;

    // Copy result back to host
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {

    int n = 43346; // Example size
    float *h_A, *h_x, *h_y;

    // Allocate host memory
    h_A = new float[n * n];
    h_x = new float[n];
    h_y = new float[n];

    



    // Initialize host arrays
    // For simplicity, initializing all elements to 1.0
    for (int i = 0; i < n * n; ++i) {
        h_A[i] = 1.0f;
    }
    for (int i = 0; i < n; ++i) {
        h_x[i] = 1.0f;
    }

    // Warmup run to avoid measuring initialization overhead
    gemv_parallel(n, h_A, h_x, h_y);

    

    // Timing 10 runs and calculating average time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        gemv_parallel(n, h_A, h_x, h_y);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    double avgTime = duration.count() / 10.0;

    // print total time
    std::cout << "Average time taken for 10 runs: " << total_time / 10 << " ms" << std::endl;

    // Free host memory
    delete[] h_A;
    delete[] h_x;
    delete[] h_y;

    return 0;
}