#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 975 // Grid size
#define NITER 10 // Number of iterations
#define NRUNS 10 // Number of runs for timing

// Global variables to measure time and total time
double total_times = 0.0;
double warmup = 0;


// Declare the jacobi_parallel function
extern "C" void jacobi_parallel(double *h_u, double *h_unew, int n, int niter);

// Function to get current time in seconds
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


__global__ void jacobi_kernel(double *u, double *unew, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int k = blockDim.z * blockIdx.z + threadIdx.z + 1;

    if (i < n - 1 && j < n - 1 && k < n - 1) {
        int idx = i + j * n + k * n * n;
        unew[idx] = 0.125 * (u[idx - 1] + u[idx + 1]
                           + u[idx - n] + u[idx + n]
                           + u[idx - n * n] + u[idx + n * n]
                           + u[idx]);
    }
}

extern "C" void jacobi_parallel(double *h_u, double *h_unew, int n, int niter) {
    double *d_u, *d_unew;
    size_t size = n * n * n * sizeof(double);

    // Allocate device memory
    cudaMalloc((void **)&d_u, size);
    cudaMalloc((void **)&d_unew, size);

    // Copy host memory to device memory
    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_unew, h_unew, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (n + threadsPerBlock.z - 1) / threadsPerBlock.z);

    double start_time = get_time();
    // Perform Jacobi iterations
    for (int iter = 0; iter < niter; iter++) {
        jacobi_kernel<<<numBlocks, threadsPerBlock>>>(d_u, d_unew, n);
        cudaDeviceSynchronize();

        // Swap pointers
        double *tmp = d_u;
        d_u = d_unew;
        d_unew = tmp;
    }
    double end_time = get_time();
    double run_time = end_time - start_time;

    // Print run time
    printf("Run time: %.6f seconds\n", run_time);

    if (warmup == 0){
        warmup = run_time;
    }
    total_times += run_time;

    // Copy device memory back to host memory
    cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_u);
    cudaFree(d_unew);
}

int main() {
    int n = N;
    size_t size = n * n * n * sizeof(double);
    double *u = (double*)malloc(size);
    double *unew = (double*)malloc(size);

    // Initialize u with random values
    for (int i = 0; i < n * n * n; i++) {
        u[i] = (double)rand() / RAND_MAX;
    }

    // Warmup run
    printf("Performing warmup run...\n");
    jacobi_parallel(u, unew, n, NITER);

    // Timing runs
    double total_time = 0.0;
    printf("Performing %d timed runs:\n", NRUNS);
    for (int run = 0; run < NRUNS; run++) {
        // Reset u to initial values for each run
        for (int i = 0; i < n * n * n; i++) {
            u[i] = (double)rand() / RAND_MAX;
        }

        double start_time = get_time();
        jacobi_parallel(u, unew, n, NITER);
        cudaDeviceSynchronize(); // Ensure GPU work is complete
        double end_time = get_time();
        double run_time = end_time - start_time;
        
        printf("Run %d: %.6f seconds\n", run + 1, run_time);
        total_time += run_time;
    }

    // Calculate and print average time
    double avg_time = total_time / NRUNS;
    printf("\nAverage time per run: %.6f seconds\n", avg_time);

    // Print total_times without warmup and average per run also without warmup
    printf("Average time per run without warmup: %.6f seconds\n", (total_times - warmup) / NRUNS);

    // Clean up
    free(u);
    free(unew);

    return 0;
}