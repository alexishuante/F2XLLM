#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void saxpy_parallel(int n, float a, float *x, float *y) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int n = 5000000; // Size of the vectors
    float a = 2.0f; // Scalar value for saxpy operation
    float *x, *y;
    double startTime, endTime, totalTime = 0, averageTime;

    // Allocate memory for vectors x and y
    x = (float*)malloc(n * sizeof(float));
    y = (float*)malloc(n * sizeof(float));

    // Initialize vectors x and y with some values
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Warmup run to ensure fair timing
    saxpy_parallel(n, a, x, y);

    // 10 consecutive runs
    for (int i = 0; i < 10; i++) {
        // Reset vector y to initial values for each run
        for (int j = 0; j < n; j++) {
            y[j] = 2.0f;
        }

        startTime = omp_get_wtime();
        saxpy_parallel(n, a, x, y);
        endTime = omp_get_wtime();

        totalTime += (endTime - startTime);
    }

    averageTime = totalTime / 10;
    printf("Average time taken for saxpy_parallel over 10 runs: %f seconds\n", averageTime);

    // Free allocated memory
    free(x);
    free(y);

    return 0;
}