#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>


void jacobi_parallel(double ***u, double ***unew, int n, int niter, int nthreads) {
    int i, j, k, iter;
    omp_set_num_threads(nthreads);

    #pragma omp parallel private(i, j, k, iter)
    {
        for (iter = 1; iter <= niter; ++iter) {
            #pragma omp for schedule(static) collapse(3)
            for (k = 1; k < n - 1; ++k) {
                for (j = 1; j < n - 1; ++j) {
                    for (i = 1; i < n - 1; ++i) {
                        unew[i][j][k] = 0.125 * (u[i - 1][j][k] + u[i + 1][j][k] +
                                                 u[i][j - 1][k] + u[i][j + 1][k] + 
                                                 u[i][j][k - 1] + u[i][j][k + 1] + 
                                                 u[i][j][k]);
                    }
                }
            }

            #pragma omp barrier

            /* Swap the arrays: u <- unew */
            #pragma omp for schedule(static) collapse(3)
            for (k = 0; k < n; ++k) {
                for (j = 0; j < n; ++j) {
                    for (i = 0; i < n; ++i) {
                        u[i][j][k] = unew[i][j][k];
                    }
                }
            }

            #pragma omp barrier
        }
    }
}

// Function to allocate 3D array
double*** allocate_3d_array(int n) {
    double ***arr = (double***)malloc(n * sizeof(double**));
    for (int i = 0; i < n; i++) {
        arr[i] = (double**)malloc(n * sizeof(double*));
        for (int j = 0; j < n; j++) {
            arr[i][j] = (double*)malloc(n * sizeof(double));
        }
    }
    return arr;
}

// Function to free 3D array
void free_3d_array(double ***arr, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            free(arr[i][j]);
        }
        free(arr[i]);
    }
    free(arr);
}

int main() {
    int n = 975;  // Size of the 3D grid
    int niter = 10;  // Number of iterations
    int nthreads = 64;  // Number of threads
    int num_runs = 10;  // Number of test runs
    
    // Allocate memory for u and unew
    double ***u = allocate_3d_array(n);
    double ***unew = allocate_3d_array(n);

    // Initialize u (you may want to set specific initial conditions)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                u[i][j][k] = 0.0;
            }
        }
    }
    
    // Set the number of threads
    omp_set_num_threads(nthreads);
    
    printf("Number of threads set to: %d\n", omp_get_max_threads());


    // Warmup run
    printf("Performing warmup run...\n");
    jacobi_parallel(u, unew, n, niter, nthreads);

    // Perform 10 test runs
    double total_time = 0.0;
    double run_times[num_runs];

    printf("Performing %d test runs...\n", num_runs);
    for (int run = 0; run < num_runs; run++) {
        double start_time = omp_get_wtime();
        
        jacobi_parallel(u, unew, n, niter, nthreads);
        
        double end_time = omp_get_wtime();
        double run_time = end_time - start_time;
        
        run_times[run] = run_time;
        total_time += run_time;
        
        printf("Run %d: %.6f seconds\n", run + 1, run_time);
    }

    double average_time = total_time / num_runs;
    printf("Average execution time: %.6f seconds\n", average_time);

    // Free allocated memory
    free_3d_array(u, n);
    free_3d_array(unew, n);

    return 0;
}
