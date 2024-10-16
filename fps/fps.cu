#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include "fps.hpp"
#include <random>



#define MAX_SAMPLES 1024

__global__ void farthestPointSamplingKernel(
    Point* points, 
    int numPoints, 
    size_t* sampledIndices, 
    int numSamples,
    float* mindist) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPoints) return;
    
    //__shared__ float minDists[10240];
    //extern __shared__ float minDists[];
    __shared__ int farthestPoint;
    
    if (threadIdx.x == 0) {
        farthestPoint = 0;
        sampledIndices[0] = farthestPoint;
    }
    
    __syncthreads();
    
    Point myPoint = points[tid];
    float myDist = FLT_MAX;
    
    for (size_t i = 1; i < numSamples; i++) {
        Point centroid = points[farthestPoint];
        float dist = (myPoint.x - centroid.x) * (myPoint.x - centroid.x) +
                     (myPoint.y - centroid.y) * (myPoint.y - centroid.y) +
                     (myPoint.z - centroid.z) * (myPoint.z - centroid.z);
        
        if (dist < myDist) {
            mindist[tid] = dist;
            myDist = dist;
        }
        
        __syncthreads();
        
        if (threadIdx.x == 0) {
            float maxDist = -FLT_MAX;
            size_t maxIdx = 0;
            
            for (size_t j = 0; j < numPoints; j++) {
                if (mindist[j] > maxDist) {
                    maxDist = mindist[j];
                    maxIdx = j;
                }
            }
            
            farthestPoint = maxIdx;
            sampledIndices[i] = farthestPoint;
        }
        
        __syncthreads();
    }
}


std::vector<size_t> farthestPointSamplingV2(const std::vector<Point>& points, int numSamples)
{
    int numPoints = points.size();

    // 分配设备内存
    thrust::device_vector<Point> d_points = points;
    thrust::device_vector<size_t> d_sampledIndices(numSamples);

    int blockSize = 1024;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;
    
    float* mindists = nullptr;
    cudaMalloc(&mindists, sizeof(float) * numPoints);
    farthestPointSamplingKernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_points.data()),
        numPoints, 
        thrust::raw_pointer_cast(d_sampledIndices.data()), 
        numSamples,
        mindists);
    
    cudaDeviceSynchronize();

    cudaFree(mindists);
    // 将结果复制回主机
    std::vector<size_t> sampledIndices(numSamples);
    try
    {
        thrust::copy(d_sampledIndices.begin(), d_sampledIndices.end(), sampledIndices.begin());
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return sampledIndices;
}




// CUDA核函数：计算每个点到当前采样集的最小距离
__global__ void computeDistances(Point* points, int numPoints, int* sampledIndices, 
                                 int numSamples, float* minDistances) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    Point p = points[idx];
    float minDist = INFINITY;

    for (int i = 0; i < numSamples; ++i) {
        Point sp = points[sampledIndices[i]];
        float dx = p.x - sp.x;
        float dy = p.y - sp.y;
        float dz = p.z - sp.z;
        float dist = dx*dx + dy*dy + dz*dz;
        minDist = min(minDist, dist);
    }

    minDistances[idx] = minDist;
}


// 主机函数：执行FPS
std::vector<int> farthestPointSampling(const std::vector<Point>& points, int numSamples) {
    int numPoints = points.size();

    // 分配设备内存
    thrust::device_vector<Point> d_points = points;
    thrust::device_vector<int> d_sampledIndices(numSamples);
    thrust::device_vector<float> d_minDistances(numPoints);

    // 随机选择第一个点
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, numPoints - 1);
    int firstIndex = dis(gen);
    d_sampledIndices[0] = firstIndex;

    // 设置CUDA核函数的参数
    int threadsPerBlock = 1024;
    int numBlocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    // 迭代选择剩余的点
    for (int i = 1; i < numSamples; ++i) {
        // 计算距离
        computeDistances<<<numBlocks, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_points.data()),
            numPoints,
            thrust::raw_pointer_cast(d_sampledIndices.data()),
            i,
            thrust::raw_pointer_cast(d_minDistances.data())
        );

        // 找到最远的点
        thrust::device_vector<float>::iterator iter = thrust::max_element(d_minDistances.begin(), d_minDistances.end());
        int farthestIndex = iter - d_minDistances.begin();
        d_sampledIndices[i] = farthestIndex;
    }

    // 将结果复制回主机
    std::vector<int> sampledIndices(numSamples);
    thrust::copy(d_sampledIndices.begin(), d_sampledIndices.end(), sampledIndices.begin());

    return sampledIndices;
}





void print_cuda_info()
{
    int device = 0; // 选择GPU设备号
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / maxThreadsPerBlock;
    int totalThreads = maxBlocksPerSM * prop.multiProcessorCount * maxThreadsPerBlock;

    std::cout << "Total threads: " << totalThreads << std::endl;
}