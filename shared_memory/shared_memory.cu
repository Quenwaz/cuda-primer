
#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel_function(int N) {
    __shared__ float sharedArray[64];  // ��̬�����ڴ�
    //extern __shared__ float sharedArray[];  // ��̬�����ڴ�
    int idx = blockIdx.x * blockDim.x +  threadIdx.x;

    // ��ʼ�������ڴ�
    if (idx < N) {
        sharedArray[idx] = (float)idx;
    }
    __syncthreads();  // ȷ�������̶߳������д����
    // ���ʹ����ڴ��е�ֵ
    if (idx < N) {
        printf("[%d] sharedArray[%d] = %f\n", blockIdx.x, idx, sharedArray[idx]);
    }
}

int main() {
    int N = 64;
    int sharedMemSize = N * sizeof(float);  // ��̬�����ڴ��С

    kernel_function << <4, 16, sharedMemSize >> > (64);  // �����˺���
    cudaDeviceSynchronize();  // �ȴ��˺������

    return 0;
}
