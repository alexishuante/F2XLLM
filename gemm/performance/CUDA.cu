#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


// Global variables used to measure the time taken by the GEMM kernel and total time
cudaEvent_t start, stop;
float elapsed_time_ms;
float kernel_time = 0.0;

float warmup = 0.0;




__global__ void gemm_kernel(int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        double temp = 0.0;
        for (int l = 0; l < k; ++l) {
            temp += a[row * lda + l] * b[l * ldb + col];
        }
        c[row * ldc + col] = alpha * temp + beta * c[row * ldc + col];
    }
}

// Host function to call the GEMM CUDA kernel
void gemm_cuda(int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {
    // Size in bytes of the matrices
    size_t size_a = m * k * sizeof(double);
    size_t size_b = k * n * sizeof(double);
    size_t size_c = m * n * sizeof(double);

    // Device pointers
    double *d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc((void**)&d_a, size_a);
    cudaMalloc((void**)&d_b, size_b);
    cudaMalloc((void**)&d_c, size_c);

    // Copy matrices to device
    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size_c, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch the GEMM kernel
    gemm_kernel<<<numBlocks, threadsPerBlock>>>(m, n, k, alpha, d_a, lda, d_b, ldb, beta, d_c, ldc);

    // Record the stop event
    cudaEventRecord(stop);
        // Synchronize and check for any errors
    cudaDeviceSynchronize();


    // Wait for the stop event
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    // Print the elapsed time
    // printf("Elapsed time: %f ms\n", elapsed_time_ms);
        // Find warmup time
    if (warmup == 0.0) {
        warmup = elapsed_time_ms;
        printf("Warmup time: %f ms\n", warmup);
    }

    // Add the time taken by the kernel to the total time only if it is not the warmup run
    if (warmup != elapsed_time_ms) {
        kernel_time += elapsed_time_ms;
        printf("Kernel time: %f ms\n", elapsed_time_ms);
    }





    // Copy result back to host
    cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}


// Main function including warmup and 10 runs to find average time taken for each run
int main() {
    // Matrix dimensions
    int m = 25027;
    int n = 25027;
    int k = 25027;

    // Alpha and beta values
    double alpha = 1.0;
    double beta = 1.0;

    // Leading dimensions
    int lda = k;
    int ldb = n;
    int ldc = n;

    // Allocate memory for matrices
    double *a = (double*)malloc(m * k * sizeof(double));
    double *b = (double*)malloc(k * n * sizeof(double));
    double *c = (double*)malloc(m * n * sizeof(double));

    // Initialize matrices
    for (int i = 0; i < m * k; ++i) {
        a[i] = 1.0;
    }
    for (int i = 0; i < k * n; ++i) {
        b[i] = 1.0;
    }
    for (int i = 0; i < m * n; ++i) {
        c[i] = 0.0;
    }

        // Print all arguments
    printf("m: %d\n", m);
    printf("n: %d\n", n);
    printf("k: %d\n", k);
    printf("alpha: %f\n", alpha);
    printf("beta: %f\n", beta);
    printf("lda: %d\n", lda);
    printf("ldb: %d\n", ldb);
    printf("ldc: %d\n", ldc);

    // Warmup
    gemm_cuda(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

    // 10 runs
    for (int i = 0; i < 10; ++i) {
        gemm_cuda(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }



    // Total time taken
    printf("Total time: %f ms\n", elapsed_time_ms);
    // Average time taken
    printf("Average time: %f ms\n", elapsed_time_ms / 10);

    // Print total_times without warmup and average per run also without warmup
    printf("Average time per run without warmup: %f ms\n", (kernel_time) / 10);




    // Free memory
    free(a);
    free(b);
    free(c);

    return 0;
}
