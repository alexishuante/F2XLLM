#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>
#include <time.h> // Include this header


void gemv_parallel(int n, float* A, float* x, float* y) {
    int i, j;
    float sum;

    #pragma acc parallel loop private(j, sum) present(A[0:n*n], x[0:n], y[0:n])
    for (i = 0; i < n; i++) {
        sum = 0.0;
        #pragma acc loop reduction(+:sum)
        for (j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

int main() {
    int n = 43346; // Size of the matrix and vectors
    float *A, *x, *y;
    clock_t start, end;
    double totalTime = 0, avgTime;

    // Allocate memory
    A = (float*)malloc(n * n * sizeof(float));
    x = (float*)malloc(n * sizeof(float));
    y = (float*)malloc(n * sizeof(float));

    // Initialize A and x with some values
    for (int i = 0; i < n; i++) {
        x[i] = 1.0; // Example initialization
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (i + j) % 100; // Example initialization
        }
    }

    // Warmup run
    gemv_parallel(n, A, x, y);

    // Timing runs
    for (int i = 0; i < 10; i++) {
        start = clock();
        gemv_parallel(n, A, x, y);
        end = clock();
        totalTime += (double)(end - start) / CLOCKS_PER_SEC;
    }

    avgTime = totalTime / 10;
    printf("Average time taken for 10 runs: %f seconds\n", avgTime);

    // Cleanup
    free(A);
    free(x);
    free(y);

    return 0;
}