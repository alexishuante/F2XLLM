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
    printf("hello\n");
    // Test code here, e.g., initialize arrays x and y, call saxpy_parallel, and print results
    return 0;
}