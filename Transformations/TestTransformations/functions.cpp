Code Block 1:
void gemv_parallel(int n, const float *A, const float *x, float *y) {
    int i, j;
    #pragma omp parallel for private(j)
    for (i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (j = 0; j < n; ++j) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

Code Block 1:
void gemv_parallel(int n, const float *A, const float *x, float *y) {
    // Parallelize the outer loop with OpenMP
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += A[i*n + j] * x[j];
        }
        y[i] = sum;
    }
}


