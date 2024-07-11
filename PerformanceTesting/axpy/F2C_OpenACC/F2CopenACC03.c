#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

void saxpy_parallel(int n, float a, const float *x, float *y) {
    int i;
    #pragma acc parallel loop present(x, y)
    for (i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int n = 925000000; // Example size
    float a = 2.0;
    float *x = (float*)malloc(n * sizeof(float));
    float *y = (float*)malloc(n * sizeof(float));

    // Initialize arrays with some values
    for(int i = 0; i < n; i++) {
        x[i] = i;
        y[i] = i;
    }

    // Warmup
    saxpy_parallel(n, a, x, y);

    clock_t start, end;
    double cpu_time_used;
    start = clock();

    for(int i = 0; i < 10; i++) {
        saxpy_parallel(n, a, x, y);
    }

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC / 10; // Average time per run

    printf("Average time taken: %f seconds\n", cpu_time_used);

    free(x);
    free(y);

    return 0;
}