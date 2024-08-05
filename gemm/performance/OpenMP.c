#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>  // for timing

void gemm_parallel(int m, 
                   int n, 
                   int k, 
                   double alpha, 
                   const double *a, 
                   int lda, 
                   const double *b, 
                   int ldb, 
                   double beta, 
                   double *c, 
                   int ldc) 
{
    int i, j, l;
    double temp;

    #pragma omp parallel for private(i, j, l, temp) schedule(static)
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            temp = 0.0;
            for (l = 0; l < k; l++) {
                temp += a[i * lda + l] * b[l * ldb + j];
            }
            c[i * ldc + j] = alpha * temp + beta * c[i * ldc + j];
        }
    }
}

int main() {
    int m = 25027, n = 25027, k = 25027;
    double alpha = 1.0, beta = 1.0;
    double *a = (double *)malloc(m * k * sizeof(double));
    double *b = (double *)malloc(k * n * sizeof(double));
    double *c = (double *)malloc(m * n * sizeof(double));

    omp_set_num_threads(64);

    int maxThreads = omp_get_max_threads();
    printf("Threads set to: %d\n", maxThreads);

    // Initialize matrices A and B
    for (int i = 0; i < m * k; i++) {
        a[i] = (double)(i % 100) / 100.0;
    }
    for (int i = 0; i < k * n; i++) {
        b[i] = (double)(i % 100) / 100.0;
    }

    // Warm-up
    gemm_parallel(m, n, k, alpha, a, k, b, n, beta, c, n);

    // Timing 10 runs and calculating average time
    clock_t start, end;
    double total_time = 0.0;
    int runs = 10;

    for (int i = 0; i < runs; i++) {
    double start_time = omp_get_wtime();
    gemm_parallel(m, n, k, alpha, a, k, b, n, beta, c, n);
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    total_time += elapsed_time;
    printf("Run %d: %f seconds\n", i, elapsed_time);
}

double average_time = total_time / runs;
printf("Average time taken for %d runs: %f seconds\n", runs, average_time);

free(a);
free(b);
free(c);

return 0;
}