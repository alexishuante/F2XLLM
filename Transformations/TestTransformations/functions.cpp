Code Block 9:
void saxpy_parallel(int n, float a, float *x, float *y) {
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

void saxpy_parallel(int n, float a, const float* x, float* y) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

void saxpy_parallel(int n, float a, const float* x, float* y) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

