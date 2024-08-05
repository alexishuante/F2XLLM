#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void gemv_parallel(int n, const float* A, const float* x, float* y) {
    int i, j;
    
    #pragma omp parallel for private(j)
    for (i = 0; i < n; ++i) {
        float sum = 0.0;
        for (j = 0; j < n; ++j) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

int main() {
    int n = 43346; // Example size
    float *A = malloc(n * n * sizeof(float));
    float *x = malloc(n * sizeof(float));
    float *y = malloc(n * sizeof(float));

    omp_set_num_threads(64);

    int maxThreads = omp_get_max_threads();
    printf("Threads set to: %d\n", maxThreads);

    // Initialize A and x with some values
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0; // Example initialization
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = (i + j) % 100; // Example initialization
        }
    }

    // Warm-up run
    gemv_parallel(n, A, x, y);

    double start, end, total_time = 0.0;
    int runs = 10;

    for (int i = 0; i < runs; ++i) {
        start = omp_get_wtime();
        gemv_parallel(n, A, x, y);
        end = omp_get_wtime();
        total_time += (end - start);
    }

    printf("Average execution time over %d runs: %f seconds\n", runs, total_time / runs);

    free(A);
    free(x);
    free(y);

    return 0;
}