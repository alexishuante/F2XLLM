#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>


__global__ void saxpy_parallel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

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


extern void saxpy(int n, float a, float *x, float *y);

int main() {
    int n = 1000000; // Size of the arrays
    float a = 2.0f; // Scalar value for saxpy
    float *x, *y;

    // Allocate memory for x and y on the host
    x = (float*)malloc(n * sizeof(float));
    y = (float*)malloc(n * sizeof(float));

    // Initialize x and y arrays
    for(int i = 0; i < n; i++) {
        x[i] = 1.0f; // Example value
        y[i] = 2.0f; // Example value
    }

    // Warmup run
    saxpy(n, a, x, y);

    // Timing
    clock_t start, end;
    double total_time = 0.0;

    for(int i = 0; i < 10; i++) {
        start = clock();
        saxpy(n, a, x, y);
        end = clock();
        total_time += (double)(end - start) / CLOCKS_PER_SEC;
    }

    double average_time = total_time / 10;
    printf("Average Time: %f seconds\n", average_time);

    // Free allocated memory
    free(x);
    free(y);

    return 0;
}