
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>

void gemv_parallel(int n, const float* A, const float* x, float* y) {
    // Parallel region with private variable sum for each thread
    #pragma omp parallel for private(j, sum)
    for (int i = 0; i < n; ++i) {
        float sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}


int main() {
    const int n = 15;
    float A[n][n], x[n], y[n];

    // Initialize the matrix and vectors
    for (int i = 0; i < n; ++i) {
        x[i] = 5.0; // Initialize x to 5.0
        y[i] = 0.0; // Initialize y to 0.0
        for (int j = 0; j < n; ++j) {
            A[i][j] = 3.0; // Initialize A to 3.0
        }
    }

    // Call the gemv_parallel function

gemv_parallel(n, (const float *)A, x, y);

    // Print the results
    for (int i = 0; i < n; ++i) {
        std::cout << "y(" << i + 1 << ") = " << y[i] << std::endl;
    }

    return 0;
}

