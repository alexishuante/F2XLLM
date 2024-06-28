#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

    void set_device(int device_id) {
        hipSetDevice(device_id);
    }

    void copy_to_device(void* src, void* dst, size_t count) {
        hipMemcpy(dst, src, count, hipMemcpyHostToDevice);
    }
    void launch_vector_add_kernel(float *A, float *B, float *C, int N);

    __global__ void vector_add_kernel(float *A, float *B, float *C, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) {
            C[i] = A[i] + B[i];
        }
    }

    void launch_vector_add_kernel(float *A, float *B, float *C, int N) {
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        hipLaunchKernelGGL(vector_add_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, A, B, C, N);
    }

#ifdef __cplusplus
}
#endif
