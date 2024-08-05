#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>


void spmv_parallel(int n, int nnz, const float* val, const int* row, const int* col, const float* x, float* y) 
{
    #pragma acc parallel loop present(val[0:nnz], row[0:n+1], col[0:nnz], x[0:n], y[0:n])
    for (int i = 0; i < n; ++i) 
    {
        y[i] = 0.0f;
        #pragma acc loop seq
        for (int j = row[i]; j < row[i+1]; ++j) 
        {
            y[i] += val[j] * x[col[j]];
        }
    }
}

// Function to get current time in seconds
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}



int main() {
    const int n = 353851;  // Number of rows
    const int nnz = 1251924838;  // Number of non-zero elements
    const int nruns = 10;  // Number of runs for timing

    float *val = (float*)malloc(nnz * sizeof(float));
    int *row = (int*)malloc((n + 1) * sizeof(int));
    int *col = (int*)malloc(nnz * sizeof(int));
    float *x = (float*)malloc(n * sizeof(float));
    float *y = (float*)malloc(n * sizeof(float));


    // Generate random input vector x
    for (int i = 0; i < n; ++i) {
        x[i] = (float)rand() / RAND_MAX;
    }



    printf("Number of rows: %d\n", n);
    printf("Number of non-zero elements: %d\n", nnz);


    // Read from file and store to arrays
    FILE* file = fopen("/home/jvalglz/gptFortranLara/spmv/matrix_csr_test_final.txt", "r");
    if (file == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    
    // Initialize counters for each array
    int val_count = 0;
    int row_count = 0;
    int col_count = 0;

    // what to expect data in file to look like
    // # Values:
    // 1.0
    // 2.0
    // 3.0
    // ...
    // # Column Indices:
    // 1
    // 2
    // 3
    // ...
    // # Row Pointers:
    // 0
    // 3
    // 6
    // ...

    // Read file line by line
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        // Check if line is a comment
        if (line[0] == '#') {
            continue;
        }

        // Read values
        if (val_count < nnz) {
            val[val_count] = atof(line);
            val_count++;
        }
        // Read column indices
        else if (col_count < nnz) {
            col[col_count] = atoi(line);
            col_count++;
        }
        // Read row pointers
        else if (row_count < n + 1) {
            row[row_count] = atoi(line);
            row_count++;
        }
    }

        // Copy data to the device
    #pragma acc enter data copyin(val[0:nnz], row[0:n+1], col[0:nnz], x[0:n], y[0:n])


    // Print counters
    printf("val_count: %d\n", val_count);
    printf("row_count: %d\n", row_count);
    printf("col_count: %d\n", col_count);

    fclose(file);

    // Print first and last elements of each array
    printf("val[0]: %f\n", val[0]);
    printf("val[%d]: %f\n", nnz - 1, val[nnz - 1]);
    printf("col[0]: %d\n", col[0]);
    printf("col[%d]: %d\n", nnz - 1, col[nnz - 1]);
    printf("row[0]: %d\n", row[0]);
    printf("row[%d]: %d\n", n, row[n]);


    // Warmup run
    printf("Performing warmup run...\n");
    spmv_parallel(n, nnz, val, row, col, x, y);

    // Timing runs
    double total_time = 0.0;
    printf("Performing %d timed runs:\n", nruns);
    for (int run = 0; run < nruns; ++run) {
        double start_time = get_time();
        spmv_parallel(n, nnz, val, row, col, x, y);
        #pragma acc wait
        double end_time = get_time();
        double run_time = end_time - start_time;
        
        printf("Run %d: %.6f seconds\n", run + 1, run_time);
        total_time += run_time;
    }

    // Calculate and print average time
    double avg_time = total_time / nruns;
    printf("\nAverage time per run: %.6f seconds\n", avg_time);

    // Print some statistics
    printf("Matrix size: %d x %d\n", n, n);
    printf("Number of non-zero elements: %d\n", nnz);

    // Copy result back to host
    #pragma acc exit data copyout(y[0:n])

    // Clean up
    // #pragma acc exit data delete(val[0:nnz], row[0:n+1], col[0:nnz], x[0:n], y[0:n])
    free(val);
    free(row);
    free(col);
    free(x);
    free(y);

    return 0;
}