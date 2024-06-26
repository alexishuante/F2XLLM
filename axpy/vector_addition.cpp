#include <hip/hip_runtime.h>
#include <iostream>

#define N 1024

// HIP kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    float *A, *B, *C; // Host vectors
    float *d_A, *d_B, *d_C; // Device vectors
    size_t size = N * sizeof(float);

    // Set the active device to device 1
    hipSetDevice(1);

    // Allocate host memory
    A = (float *)malloc(size);
    B = (float *)malloc(size);
    C = (float *)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    hipMalloc((void **)&d_A, size);
    hipMalloc((void **)&d_B, size);
    hipMalloc((void **)&d_C, size);

    // Copy vectors from host to device
    hipMemcpy(d_A, A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, size, hipMemcpyHostToDevice);

    // Perform the vector addition 10 times
    for (int iter = 0; iter < 1000; ++iter) {
        // Launch the Vector Add HIP Kernel
        int threadsPerBlock = 256;
        int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;
        hipLaunchKernelGGL(vectorAdd, blocksPerGrid, threadsPerBlock, 0, 0, d_A, d_B, d_C, N);

        // Copy result back to host after each addition
        hipMemcpy(C, d_C, size, hipMemcpyDeviceToHost);
    }

    // Cleanup
    free(A);
    free(B);
    free(C);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}
