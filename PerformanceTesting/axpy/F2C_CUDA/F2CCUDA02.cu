// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda_runtime.h>
// #include <time.h>

// __global__ void saxpy(int n, float a, float *x, float *y) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) {
//         y[i] = a * x[i] + y[i];
//     }
// }

// void saxpy_parallel(int n, float a, float *x, float *y) {
//     // Allocate device memory for x and y
//     float *d_x, *d_y;
//     cudaMalloc(&d_x, n * sizeof(float));
//     cudaMalloc(&d_y, n * sizeof(float));

//     // Copy vectors x and y to device
//     cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

//     // Configure launch parameters
//     int blockSize = 256;
//     int numBlocks = (n + blockSize - 1) / blockSize;

//     // Launch kernel on the device
//     saxpy<<<numBlocks, blockSize>>>(n, a, d_x, d_y);

//     // Copy result back to host
//     cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_x);
//     cudaFree(d_y);
// }




// int main() {
//     int n = 1000000; // Size of the arrays
//     float a = 2.0f; // Scalar value for saxpy
//     float *x, *y;

//     // Allocate memory for x and y on the host
//     x = (float*)malloc(n * sizeof(float));
//     y = (float*)malloc(n * sizeof(float));

//     // Initialize x and y arrays
//     for(int i = 0; i < n; i++) {
//         x[i] = 1.0f; // Example value
//         y[i] = 2.0f; // Example value
//     }

//     // Warmup run
//     saxpy_parallel(n, a, x, y);

//     // Create CUDA events
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     float milliseconds = 0;

//     for(int i = 0; i < 10; i++) {
//         cudaEventRecord(start); // Record start event
//         saxpy_parallel(n, a, x, y); // Kernel execution
//         cudaEventRecord(stop); // Record stop event

//         cudaEventSynchronize(stop); // Wait for the event to be completed
//         float millisecondsTemp = 0;
//         cudaEventElapsedTime(&millisecondsTemp, start, stop); // Calculate elapsed time
//         milliseconds += millisecondsTemp;
//     }

//     // Print average time
//     printf("Average time: %f milliseconds\n", milliseconds / 10);

//     // Destroy CUDA events
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
// }

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

void saxpy_parallel(int n, float a, float *d_x, float *d_y) { // Note: Taking device pointers
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    saxpy<<<numBlocks, blockSize>>>(n, a, d_x, d_y); 
}


int main() {
    // ... (array initialization remains the same)
    int n = 1000000; // Size of the arrays
    float a = 2.0f; // Scalar value for saxpy

    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(n * sizeof(float));
    y = (float*)malloc(n * sizeof(float));

    // Initialize x and y arrays
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    // Copy data to device (outside timing)
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup run (outside timing)
    saxpy_parallel(n, a, d_x, d_y);
    cudaDeviceSynchronize(); 

        // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    for(int i = 0; i < 10; i++) {
        cudaEventRecord(start); // Record start event
        saxpy_parallel(n, a, d_x, d_y); 
        cudaEventRecord(stop); // Record stop event

        cudaEventSynchronize(stop); // Wait for the event to be completed
        float millisecondsTemp = 0;
        cudaEventElapsedTime(&millisecondsTemp, start, stop); // Calculate elapsed time
        milliseconds += millisecondsTemp;
    }

    // Print average time
    printf("Average time: %f milliseconds\n", milliseconds / 10);

    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost); // Copy back result (outside timing)
    
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
    
    return 0;
}

