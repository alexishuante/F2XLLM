#include <hip/hip_runtime.h>

// Define the kernel
__global__ void axpy_kernel(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// Define the wrapper function
extern "C" {
    void saxpy_wrapper(int* n, float* a, float* x, float* y) {
        float *d_x, *d_y;
        hipMalloc(&d_x, (*n) * sizeof(float));
        hipMalloc(&d_y, (*n) * sizeof(float));

        hipMemcpy(d_x, x, (*n) * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_y, y, (*n) * sizeof(float), hipMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = ((*n) + blockSize - 1) / blockSize;
        hipLaunchKernelGGL(axpy_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, *n, *a, d_x, d_y);

        hipMemcpy(y, d_y, (*n) * sizeof(float), hipMemcpyDeviceToHost);

        hipFree(d_x);
        hipFree(d_y);
    }

    void set_gpu_device(int device_id) {
        hipSetDevice(device_id);
    }
    
}
