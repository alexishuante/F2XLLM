#include <hip/hip_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16

// Global variables to measure time
hipEvent_t start, stop;
// Total time taken by the kernel
float kernel_time = 0.0;
float warmup = 0.0;


__global__ void gemm_kernel(int m, int n, int k, double alpha, const double *a, int lda, 
                            const double *b, int ldb, double beta, double *c, int ldc) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < m && Col < n) {
        double temp = 0.0;
        for (int i = 0; i < k; ++i) {
            temp += a[Row * lda + i] * b[i * ldb + Col];
        }
        c[Row * ldc + Col] = alpha * temp + beta * c[Row * ldc + Col];
    }
}

void gemm_hip(int m, int n, int k, double alpha, const double *a, int lda, 
              const double *b, int ldb, double beta, double *c, int ldc) {
    double *d_a, *d_b, *d_c;

    // Allocate device memory
    hipMalloc(&d_a, m * k * sizeof(double));
    hipMalloc(&d_b, k * n * sizeof(double));
    hipMalloc(&d_c, m * n * sizeof(double));

    // Copy matrices to device
    hipMemcpy(d_a, a, m * k * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b, b, k * n * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_c, c, m * n * sizeof(double), hipMemcpyHostToDevice);

    // Define the block and grid dimensions 
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    // Launch the kernel
    hipLaunchKernelGGL(gemm_kernel, dimGrid, dimBlock, 0, 0, m, n, k, alpha, d_a, lda, d_b, ldb, beta, d_c, ldc);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipDeviceSynchronize();


    // Calculate the time taken by the kernel
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);

    // Add the time taken by the kernel to the total time only if it is not the warmup run
    if (warmup != 0.0) {
        printf("Time taken for kernel: %f ms\n", milliseconds);
        kernel_time += milliseconds;
    }

    // Find warmup time
    if (warmup == 0.0) {
        warmup = milliseconds;
        printf("Warmup time: %f ms\n", warmup);
    }


    // Copy the result matrix back to host
    hipMemcpy(c, d_c, m * n * sizeof(double), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
}


// Main program that runs a warmup then do 10 runs of the gemm kernel
int main() {
    int m = 25027, n = 25027, k = 25027;
    double alpha = 1.0, beta = 1.0;
    double *a = (double *)malloc(m * k * sizeof(double));
    double *b = (double *)malloc(k * n * sizeof(double));
    double *c = (double *)malloc(m * n * sizeof(double));

    // Initialize matrices A and B
    for (int i = 0; i < m * k; i++) {
        a[i] = (double)(i % 100) / 100.0;
    }
    for (int i = 0; i < k * n; i++) {
        b[i] = (double)(i % 100) / 100.0;
    }

    // Warm-up
    gemm_hip(m, n, k, alpha, a, k, b, n, beta, c, n);

    // Timing 10 runs and calculating average time
    int runs = 10;
    for (int i = 0; i < runs; i++) {
        gemm_hip(m, n, k, alpha, a, k, b, n, beta, c, n);
    }

    // Print all variables
    printf("Threads set to: %d\n", 64);
    printf("m = %d, n = %d, k = %d\n", m, n, k);
    printf("alpha = %f, beta = %f\n", alpha, beta);

    // Print the average time taken by the kernel
    printf("Average time taken for kernel: %f ms\n", kernel_time / runs);

    free(a);
    free(b);
    free(c);

    return 0;
}