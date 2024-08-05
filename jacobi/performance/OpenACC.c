#include <stdio.h>
#include <stdlib.h>
#include <time.h>



void jacobi_parallel(double *u, double *unew, int n, int niter) {
    int i, j, k, iter;

    #pragma acc data copy(u[0:n*n*n], unew[0:n*n*n])
    {
        for (iter = 0; iter < niter; iter++) {
            #pragma acc parallel loop collapse(3) private(i, j, k)
            for (k = 1; k < n - 1; k++) {
                for (j = 1; j < n - 1; j++) {
                    for (i = 1; i < n - 1; i++) {
                        unew[i + n * (j + n * k)] = 0.125 * (
                            u[(i - 1) + n * (j + n * k)] +
                            u[(i + 1) + n * (j + n * k)] +
                            u[i + n * ((j - 1) + n * k)] +
                            u[i + n * ((j + 1) + n * k)] +
                            u[i + n * (j + n * (k - 1))] +
                            u[i + n * (j + n * (k + 1))] +
                            u[i + n * (j + n * k)]
                        );
                    }
                }
            }

            // Swap pointers for the next iteration
            double *temp = u;
            u = unew;
            unew = temp;
        }
    }
}

#define N 975        // Size of the 3D grid
#define NITER 10     // Number of iterations
#define NRUNS 10     // Number of runs for timing

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    int n = N;
    int size = n * n * n;
    double *u = (double*)malloc(size * sizeof(double));
    double *unew = (double*)malloc(size * sizeof(double));

    // Initialize u with random values
    for (int i = 0; i < size; i++) {
        u[i] = (double)rand() / RAND_MAX;
    }

    // Warmup run
    printf("Performing warmup run...\n");
    jacobi_parallel(u, unew, n, NITER);

    // Timing runs
    double total_time = 0.0;
    printf("Performing %d timed runs:\n", NRUNS);
    
    for (int run = 0; run < NRUNS; run++) {
        // Reset u to initial values for each run
        for (int i = 0; i < size; i++) {
            u[i] = (double)rand() / RAND_MAX;
        }

        double start_time = get_time();
        jacobi_parallel(u, unew, n, NITER);
        double end_time = get_time();
        double run_time = end_time - start_time;
        
        printf("Run %d: %.6f seconds\n", run + 1, run_time);
        total_time += run_time;
    }

    // Calculate and print average time
    double avg_time = total_time / NRUNS;
    printf("\nAverage time per run: %.6f seconds\n", avg_time);

    // Clean up
    free(u);
    free(unew);

    return 0;
}