
#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel_function(int N) {
    __shared__ float sharedArray[64];  // 静态共享内存
    //extern __shared__ float sharedArray[];  // 动态共享内存
    int idx = blockIdx.x * blockDim.x +  threadIdx.x;

    // 初始化共享内存
    if (idx < N) {
        sharedArray[idx] = (float)idx;
    }
    __syncthreads();  // 确保所有线程都完成了写操作
    // 访问共享内存中的值
    if (idx < N) {
        printf("[%d] sharedArray[%d] = %f\n", blockIdx.x, idx, sharedArray[idx]);
    }
}

int main() {
    int N = 64;
    int sharedMemSize = N * sizeof(float);  // 动态共享内存大小

    kernel_function << <4, 16, sharedMemSize >> > (64);  // 启动核函数
    cudaDeviceSynchronize();  // 等待核函数完成

    return 0;
}
