
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>

void saxpy_parallel(int n, float a, float *x, float *y) {
    int i;

    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}


int main() {
    const int n = 10;
    float x[n], y[n];
    float a = 2.0;

    // Initialize arrays x and y
    for (int i = 0; i < n; ++i) {
        x[i] = 3.0;
        y[i] = 0.0;
    }

    // Call the saxpy_parallel function
    saxpy_parallel(n, a, x, y);

    // Print the results
    for (int i = 0; i < n; ++i) {
        std::cout << "y(" << i + 1 << ") = " << y[i] << std::endl;
    }
    return 0;
}
