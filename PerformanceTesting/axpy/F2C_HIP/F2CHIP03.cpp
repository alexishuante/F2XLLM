#include <iostream>
#include <hip/hip_runtime.h>

__global__ void saxpy_parallel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int n = 1000000; // Size of vectors
    float a = 2.0f; // Scalar value
    float *x, *y, *d_x, *d_y;
    float totalTime = 0.0f, averageTime;
    hipEvent_t start, stop;

    // Allocate host memory
    x = new float[n];
    y = new float[n];

    // Initialize host arrays
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Allocate device memory
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));

    // Copy host vectors to device
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    // Create events
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Warm-up run
    saxpy_parallel<<<(n + 255) / 256, 256>>>(n, a, d_x, d_y);

    // Ensure the warm-up run is completed
    hipDeviceSynchronize();

    // Timed runs
    for (int i = 0; i < 10; ++i) {
        hipEventRecord(start);
        saxpy_parallel<<<(n + 255) / 256, 256>>>(n, a, d_x, d_y);
        hipEventRecord(stop);

        hipEventSynchronize(stop);
        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;
    }

    // Calculate average time
    averageTime = totalTime / 10.0f;
    std::cout << "Average time: " << averageTime << " ms\n";

    // Cleanup
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_x);
    hipFree(d_y);
    delete[] x;
    delete[] y;

    return 0;
}