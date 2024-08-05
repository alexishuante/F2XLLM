
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define N 353851    // Number of rows in the matrix
#define NNZ 1251924838  // Number of non-zero elements
#define NRUNS 10     // Number of runs for timing


// Global variables to measure time and total time
double total_times = 0.0;
double warmup = 0;

// Function to get current time in seconds
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// CUDA Error handling macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = call;                                          \
        if (error != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__);    \
            fprintf(stderr, "code: %d, reason: %s\n", error,               \
                    cudaGetErrorString(error));                            \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// CUDA Kernel
__global__
void spmv_kernel(int n, int nnz, const float* val, const int* row, const int* col, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float sum = 0.0f;
        for (int j = row[i]; j < row[i + 1]; j++) {
            sum += val[j] * x[col[j]];
        }
        y[i] = sum;
    }
}

// Host function
void spmv_parallel(int n, int nnz, const float* h_val, const int* h_row, const int* h_col, const float* h_x, float* h_y) {
    float *d_val, *d_x, *d_y;
    int *d_row, *d_col;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_val, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_row, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_col, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, n * sizeof(float)));

    // Transfer data to device
    CUDA_CHECK(cudaMemcpy(d_val, h_val, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row, h_row, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col, h_col, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));

    // Define block and grid size
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    double start_time = get_time();
    // Launch the kernel
    spmv_kernel<<<gridSize, blockSize>>>(n, nnz, d_val, d_row, d_col, d_x, d_y);
    cudaDeviceSynchronize();
    double end_time = get_time();
    double run_time = end_time - start_time;

    // Print run time
    printf("Run time: %.6f seconds\n", run_time);

    if (warmup == 0){
        warmup = run_time;
    }
    total_times += run_time;


    // Check for any kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Transfer the result back to host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Deallocate device memory
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_row));
    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}



int main() {
    int n = N;
    int nnz = NNZ;

    printf("Number of rows: %d\n", n);
    printf("Number of non-zero elements: %d\n", nnz);


    // Allocate host memory
    float* h_val = (float*)malloc(nnz * sizeof(float));
    int* h_row = (int*)malloc((n + 1) * sizeof(int));
    int* h_col = (int*)malloc(nnz * sizeof(int));
    float* h_x = (float*)malloc(n * sizeof(float));
    float* h_y = (float*)malloc(n * sizeof(float));
    std::string filePath = "/home/jvalglz/gptFortranLara/spmv/matrix_csr_test_final.txt";
    std::vector<std::string> searchStrings = {"# Values:", "# Column Indices:", "# Row Pointers:"};

    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return 1;
    }

    // start reading the file
    std::string line;
    int i = 0;
    int j = 0;
    int k = 0;
    while (std::getline(file, line)) {
        if (line == searchStrings[0]) {
            i = 0;
            j = 0;
            k = 0;
            while (std::getline(file, line)) {
                if (line == searchStrings[1]) {
                    break;
                }
                h_val[i] = std::stof(line);
                i++;
            }
        }
        
        if (line == searchStrings[1]) {
            while (std::getline(file, line)) {
                if (line == searchStrings[2]) {
                    break;
                }
                h_col[j] = std::stoi(line);
                j++;
            }
        }
        if (line == searchStrings[2]) {
            while (std::getline(file, line)) {
                h_row[k] = std::stoi(line);
                k++;
            }
        }
        printf("i: %d, j: %d, k: %d\n", i, j, k);
    }




    // Print size of arrays
    std::cout << "Size of h_val: " << sizeof(*h_val) << std::endl;
    std::cout << "Size of h_col: " << sizeof(*h_col) << std::endl;
    std::cout << "Size of h_row: " << sizeof(*h_row) << std::endl;
    std::cout << "Size of h_x: " << sizeof(*h_x) << std::endl;
    std::cout << "Size of h_y: " << sizeof(*h_y) << std::endl;
    
    // Warmup run
    printf("Performing warmup run...\n");
    spmv_parallel(n, nnz, h_val, h_row, h_col, h_x, h_y);

    // Timing runs
    double total_time = 0.0;
    printf("Performing %d timed runs:\n", NRUNS);
    for (int run = 0; run < NRUNS; run++) {
        spmv_parallel(n, nnz, h_val, h_row, h_col, h_x, h_y);
        cudaDeviceSynchronize(); // Ensure GPU work is complete
        
        // printf("Run %d: %.6f seconds\n", run + 1, run_time);
        total_time += run_time;
    }

    // Calculate and print average time
    double avg_time = total_time / NRUNS;
    // printf("\nAverage time per run: %.6f seconds\n", avg_time);

    // Print total_times without warmup and average per run also without warmup
    printf("Average time per run without warmup: %.6f seconds\n", (total_times - warmup) / NRUNS);

    // Clean up
    free(h_val);
    free(h_row);
    free(h_col);
    free(h_x);
    free(h_y);

    return 0;
}