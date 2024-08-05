#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <time.h>

// Global variables to measure time and total time
double total_times = 0.0;
double warmup = 0;



#define N 975 // Define the grid size
#define NITER 10 // Define the number of iterations
#define NRUNS 10 // Number of runs for timing

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// HIP kernel to perform the Jacobi update
__global__ void jacobi_kernel(double *u, double *unew, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < n - 1 && j < n - 1 && k < n - 1) {
        unew[i + j * n + k * n * n] = 0.125 * (u[(i-1) + j * n + k * n * n] +
                                               u[(i+1) + j * n + k * n * n] +
                                               u[i + (j-1) * n + k * n * n] +
                                               u[i + (j+1) * n + k * n * n] +
                                               u[i + j * n + (k-1) * n * n] +
                                               u[i + j * n + (k+1) * n * n] +
                                               u[i + j * n + k * n * n]);
    }
}

void jacobi_parallel(double *u, double *unew, int n, int niter) {
    double *d_u, *d_unew;

    // Allocate device memory
    hipMalloc((void**)&d_u, n*n*n*sizeof(double));
    hipMalloc((void**)&d_unew, n*n*n*sizeof(double));

    // Copy data from host to device
    hipMemcpy(d_u, u, n*n*n*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_unew, unew, n*n*n*sizeof(double), hipMemcpyHostToDevice);

    dim3 blockDim(8, 8, 8);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y,
                 (n + blockDim.z - 1) / blockDim.z);

    double start_time = get_time();
    for (int iter = 0; iter < niter; ++iter) {


        // Launch the Jacobi kernel
        hipLaunchKernelGGL(jacobi_kernel, gridDim, blockDim, 0, 0, d_u, d_unew, n);
        hipDeviceSynchronize();

        // Swap pointers
        double *temp = d_u;
        d_u = d_unew;
        d_unew = temp;
    }

    double end_time = get_time();
    double run_time = end_time - start_time;

    // Print run time
    printf("Run time: %.6f seconds\n", run_time);

    if (warmup == 0){
        warmup = run_time;
    }
    total_times += run_time;

    // Copy data from device back to host
    hipMemcpy(u, d_u, n*n*n*sizeof(double), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_u);
    hipFree(d_unew);
}




int main() {
    int n = N;
    int size = n * n * n;
    double *u = (double*)malloc(size * sizeof(double));
    double *unew = (double*)malloc(size * sizeof(double));

    // Initialize u with random values
    for (int i = 0; i < size; i++) {
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
        for (int i = 0; i < size; i++) {
            u[i] = (double)rand() / RAND_MAX;
        }

        double start_time = get_time();
        jacobi_parallel(u, unew, n, NITER);
        hipDeviceSynchronize(); // Ensure GPU work is complete
        double end_time = get_time();
        double run_time = end_time - start_time;
        
        printf("Kernel Run %d: %.6f seconds\n", run + 1, run_time);
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