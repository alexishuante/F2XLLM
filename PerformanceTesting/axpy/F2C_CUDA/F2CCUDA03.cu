#include <stdio.h>
#include <cuda_runtime.h>

__global__ void saxpy_parallel(int n, float a, float *x, float *y) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = a * x[idx] + y[idx];
  }
}

int main() {
    int n = 1000000; // Size of vectors
    float a = 2.0; // Scalar value
    float *x, *y, *d_x, *d_y;
    float totalTime = 0.0, averageTime;
    cudaEvent_t start, stop;

    // Allocate host memory
    x = (float*)malloc(n * sizeof(float));
    y = (float*)malloc(n * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    // Copy host vectors to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    saxpy_parallel<<<(n + 255) / 256, 256>>>(n, a, d_x, d_y);

    // Ensure the warm-up run is completed
    cudaDeviceSynchronize();

    // Timed runs
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(start);
        saxpy_parallel<<<(n + 255) / 256, 256>>>(n, a, d_x, d_y);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;
    }

    // Calculate average time
    averageTime = totalTime / 10.0;
    printf("Average time: %f ms\n", averageTime);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    return 0;
}