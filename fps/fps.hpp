#ifndef __FPS_INCLUDED__
#define __FPS_INCLUDED__

#include <vector>

// �����ṹ
struct Point {
    float x, y, z;
};


std::vector<size_t> farthestPointSamplingV2(const std::vector<Point>& points, int numSamples);

void print_cuda_info();


#endif // __FPS_INCLUDED__