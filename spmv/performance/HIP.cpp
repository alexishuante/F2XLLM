#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <hip/hip_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>




// Global variables to measure time and total time
double total_times = 0.0;
double warmup = 0;

// Function to get current time in seconds
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// HIP kernel for sparse matrix-vector multiplication (SpMV)
__global__ void spmv_kernel(int n, int nnz, const float *val, const int *row, const int *col, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int j = row[i]; j < row[i + 1]; ++j) {
            sum += val[j] * x[col[j]];
        }
        y[i] = sum;
    }
}

// Function to perform SpMV on the GPU
void spmv_parallel(int n, int nnz, const float *val, const int *row, const int *col, const float *x, float *y) {
    // Device pointers
    float *d_val, *d_x, *d_y;
    int *d_row, *d_col;
    
    // Allocate memory on the GPU
    hipMalloc((void**)&d_val, nnz * sizeof(float));
    hipMalloc((void**)&d_row, (n + 1) * sizeof(int));
    hipMalloc((void**)&d_col, nnz * sizeof(int));
    hipMalloc((void**)&d_x, n * sizeof(float));
    hipMalloc((void**)&d_y, n * sizeof(float));
    
    // Copy data to the GPU
    hipMemcpy(d_val, val, nnz * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_row, row, (n + 1) * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_col, col, nnz * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    
    // Launch the kernel with appropriate block and grid sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    double start_time = get_time();
    hipLaunchKernelGGL(spmv_kernel, dim3(gridSize), dim3(blockSize), 0, 0, n, nnz, d_val, d_row, d_col, d_x, d_y);
    hipDeviceSynchronize();
    double end_time = get_time();
    double run_time = end_time - start_time;

    // Print run time
    printf("Run time: %.6f seconds\n", run_time);

    if (warmup == 0){
        warmup = run_time;
    }
    total_times += run_time;
    
    // Copy the result back to the host
    hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);
    
    // Free device memory
    hipFree(d_val);
    hipFree(d_row);
    hipFree(d_col);
    hipFree(d_x);
    hipFree(d_y);
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

    std::string filePath = "/home/jvalglz/gptFortranLara/spmv/matrix_csr_test_final.txt";
    std::ifstream
    file(filePath);

    if (!file.is_open()) {
        std::cout << "Error opening file" << std::endl;
        return 1;
    }

    std::string line;
    int i = 0, j = 0, k = 0;
    std::vector<std::string> searchStrings = {"# Values:", "# Column Indices:", "# Row Pointers:"};

    while (std::getline(file, line)) {
        if (line.compare(searchStrings[0]) == 0) {
            i = 0;
            j = 0;
            k = 0;
            while (std::getline(file, line)) {
                if (line.compare(searchStrings[1]) == 0) {
                    break;
                }
                val[i] = std::stof(line);
                i++;
            }
        }
        if (line.compare(searchStrings[1]) == 0) {
            while (std::getline(file, line)) {
                if (line.compare(searchStrings[2]) == 0) {
                    break;
                }
                col[j] = std::stoi(line);
                j++;
            }
        }
        if (line.compare(searchStrings[2]) == 0) {
            while (std::getline(file, line)) {
                row[k] = std::stoi(line);
                k++;
            }
        }
    }

    printf("i: %d\n", i);
    printf("j: %d\n", j);
    printf("k: %d\n", k);
    




    // Generate random input vector x
    for (int i = 0; i < n; ++i) {
        x[i] = (float)rand() / RAND_MAX;
    }

    // Warmup run
    printf("Performing warmup run...\n");
    spmv_parallel(n, nnz, val, row, col, x, y);

    // Timing runs
    double total_time = 0.0;
    printf("Performing %d timed runs:\n", nruns);
    for (int run = 0; run < nruns; ++run) {
        double start_time = get_time();
        spmv_parallel(n, nnz, val, row, col, x, y);
        hipDeviceSynchronize();  // Ensure GPU work is complete
        double end_time = get_time();
        double run_time = end_time - start_time;
        
        // printf("Run %d: %.6f seconds\n", run + 1, run_time);
        total_time += run_time;
    }

    // Calculate and print average time
    double avg_time = total_time / nruns;
    // printf("\nAverage time per run: %.6f seconds\n", avg_time);

    // Print some statistics
    printf("Matrix size: %d x %d\n", n, n);
    printf("Number of non-zero elements: %d\n", nnz);
    printf("Sparsity: %.2f%%\n", (1.0 - (double)nnz / (n * n)) * 100);
    printf("Average time per run without warmup: %.6f seconds\n", (total_times - warmup) / nruns);


    // Clean up
    free(val);
    free(row);
    free(col);
    free(x);
    free(y);

    return 0;
}