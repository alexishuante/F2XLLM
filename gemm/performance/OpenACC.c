#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

void gemm_parallel(int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {
    #pragma acc parallel loop collapse(2) present(a[0:lda*k], b[0:ldb*n], c[0:ldc*n])
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double temp = 0.0;
            for (int l = 0; l < k; ++l) {
                temp += a[i + l * lda] * b[l + j * ldb];
            }
            c[i + j * ldc] = alpha * temp + beta * c[i + j * ldc];
        }
    }
}
double run_test(int m, int n, int k, double alpha, double beta) {
    double *a, *b, *c;
    int lda = m, ldb = k, ldc = m;
    
    a = (double*)malloc(m * k * sizeof(double));
    b = (double*)malloc(k * n * sizeof(double));
    c = (double*)malloc(m * n * sizeof(double));

    // Initialize matrices with random values
    for (int i = 0; i < m * k; i++) a[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < k * n; i++) b[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < m * n; i++) c[i] = (double)rand() / RAND_MAX;

    // Copy data to device
    #pragma acc enter data copyin(a[0:m*k], b[0:k*n], c[0:m*n])

    clock_t start = clock();
    gemm_parallel(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    clock_t end = clock();

    // Copy data back from device
    #pragma acc exit data delete(a[0:m*k], b[0:k*n]) copyout(c[0:m*n])

    free(a);
    free(b);
    free(c);

    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

int main() {
    int m = 25027, n = 25027, k = 25027;
    double alpha = 1.0, beta = 0.0;
    int warmup_runs = 1;
    int test_runs = 10;

    // Print all the parameters
    printf("Matrix dimensions: %d x %d x %d\n", m, n, k);
    printf("Alpha: %.1f\n", alpha);
    printf("Beta: %.1f\n", beta);
    

    // Warmup runs
    printf("Performing %d warmup runs...\n", warmup_runs);
    for (int i = 0; i < warmup_runs; i++) {
        run_test(m, n, k, alpha, beta);
    }

    // Test runs
    printf("Performing %d test runs...\n", test_runs);
    double total_time = 0.0;
    for (int i = 0; i < test_runs; i++) {
        double run_time = run_test(m, n, k, alpha, beta);
        total_time += run_time;
        printf("Run %d: %.6f seconds\n", i + 1, run_time);
    }

    double average_time = total_time / test_runs;
    printf("Average execution time: %.6f seconds\n", average_time);

    return 0;
}