#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel for maximum reduction
__global__ void maxReduction(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (i < n) ? input[i] : -INFINITY;
    printf("sdata[%d] = input[%d] = %f\n", tid, i, sdata[tid]);
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        printf("[%d] sdata[%d] = fmaxf(sdata[%d], sdata[%d]) = %f\n", blockIdx.x, tid, tid, tid + s, sdata[tid]);
        __syncthreads();

        printf("\n");
    }

    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Host function to setup and launch kernel
void findMaxCUDA(float* h_input, int n, float* result) {
    float* d_input, * d_output;
    int blockSize = 4;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Allocate device memory
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, gridSize * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    maxReduction << <gridSize, blockSize, blockSize * sizeof(float) >> > (d_input, d_output, n);

    // Copy result back to host
    float* h_output = new float[gridSize];
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Find max of partial results on CPU
    *result = h_output[0];
    for (int i = 1; i < gridSize; i++) {
        *result = fmaxf(*result, h_output[i]);
    }

    // Clean up
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    const int N = 4;
    float* h_input = new float[N];

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float result;
    findMaxCUDA(h_input, N, &result);

    printf("Maximum value: %f\n", result);

    delete[] h_input;
    return 0;
}