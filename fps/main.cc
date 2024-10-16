#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <windows.h>
#include "fps.hpp"

#undef max
#undef min

struct PCDData {
    Point xyz;
    uint8_t r, g, b;
    uint8_t label;
};

#pragma pack(push, 1)
// LAS文件头结构
struct LASHeader {
    char fileSignature[4];
    uint16_t fileSourceID;
    uint16_t globalEncoding;
    uint32_t projectIDGUID1;
    uint16_t projectIDGUID2;
    uint16_t projectIDGUID3;
    uint8_t projectIDGUID4[8];
    uint8_t versionMajor;
    uint8_t versionMinor;
    char systemIdentifier[32];
    char generatingSoftware[32];
    uint16_t fileCreationDayOfYear;
    uint16_t fileCreationYear;
    uint16_t headerSize;
    uint32_t offsetToPointData;
    uint32_t numberOfVariableLengthRecords;
    uint8_t pointDataRecordFormat;
    uint16_t pointDataRecordLength;
    uint32_t numberOfPointRecords;
    uint32_t numberOfPointsByReturn[5];
    double xScaleFactor;
    double yScaleFactor;
    double zScaleFactor;
    double xOffset;
    double yOffset;
    double zOffset;
    double maxX;
    double minX;
    double maxY;
    double minY;
    double maxZ;
    double minZ;
};


struct LASPoint {
    int32_t X;
    int32_t Y;
    int32_t Z;
    uint16_t intensity;
    uint8_t returnNumber : 3;
    uint8_t numberOfReturns : 3;
    uint8_t scanDirectionFlag : 1;
    uint8_t edgeOfFlightLine : 1;
    uint8_t classification;
    int8_t scanAngleRank;
    uint8_t userData;
    uint16_t pointSourceID;
    // Add more fields if needed for different point data formats
};
#pragma pack(pop)

size_t readLasfileWithMMP(const std::string& filename, std::vector<Point>& points)
{
    const auto test = sizeof LASPoint;
    HANDLE hFile = CreateFile(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "Failed to open file" << std::endl;
        return 0;
    }

    HANDLE hMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMapping == NULL) {
        std::cerr << "Failed to create file mapping" << std::endl;
        CloseHandle(hFile);
        return 0;
    }

    LPVOID pData = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (pData == NULL) {
        std::cerr << "Failed to map view of file" << std::endl;
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return 0;
    }

    auto header = static_cast<LASHeader*>(pData);
    points.reserve(header->numberOfPointRecords);

    auto points_addr = static_cast<char*>(pData) + header->offsetToPointData;
    size_t offset = 0;
    for (uint32_t i = 0; i < header->numberOfPointRecords; ++i) {
        int32_t x = *reinterpret_cast<int32_t*>(points_addr + offset),
            y = *reinterpret_cast<int32_t*>(points_addr + offset + sizeof(int32_t)),
            z = *reinterpret_cast<int32_t*>(points_addr + offset + sizeof(int32_t) * 2);


        Point p;
        p.x = float(x * header->xScaleFactor + header->xOffset);
        p.y = float(y * header->yScaleFactor + header->yOffset);
        p.z = float(z * header->zScaleFactor + header->zOffset);

        points.push_back(p);
        offset += header->pointDataRecordLength;
    }

    UnmapViewOfFile(pData);
    pData = NULL;
    CloseHandle(hMapping);
    hMapping = NULL;
    CloseHandle(hFile);
    hFile = INVALID_HANDLE_VALUE;

    return points.size();
}

std::vector<PCDData> read_pcd_from_ascii(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Unable to open file");
    }

    std::vector<PCDData> result;
    std::string line;
    while (std::getline(ifs, line)) { 
        double x = 0, y = 0, z = 0;
        float r = 0, g = 0, b = 0;
        float label = 0;
        if (7 != std::sscanf(line.c_str(), "%lf%lf%lf%f%f%f%f", &x, &y, &z, &r, &g, &b, &label)) {
            continue;
        }
        result.emplace_back(PCDData{ Point{float(x), float(y), float(z)},uint8_t(r),uint8_t(g) ,uint8_t(b),uint8_t(label) });
    }

    ifs.close();
    return result;
}

void dump_to_ascii(const std::string& filename, const std::vector<PCDData>& points, const std::vector<size_t>& indices)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error("Unable to open file");
    }

    for (size_t i = 0; i < indices.size(); ++i)
    {
        auto & pcddata = points.at(i);
        char buf[1024] = { 0 };
        std::sprintf(buf, "%lf %lf %lf %d %d %d %d\n", 
            pcddata.xyz.x, pcddata.xyz.y, pcddata.xyz.z, pcddata.r, pcddata.g, pcddata.b, pcddata.label);
        ofs << buf;
    }
    ofs.close();
}


std::vector<Point> readLASFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open file");
    }

    LASHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(LASHeader));

    // 检查文件签名
    if (std::string(header.fileSignature, 4) != "LASF") {
        throw std::runtime_error("Invalid LAS file");
    }

    // 移动到点数据开始位置
    file.seekg(header.offsetToPointData);

    std::vector<Point> points;
    points.reserve(header.numberOfPointRecords);

    for (uint32_t i = 0; i < header.numberOfPointRecords; ++i) {
        int32_t x, y, z;
        file.read(reinterpret_cast<char*>(&x), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&y), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&z), sizeof(int32_t));

        Point p;
        p.x = float(x * header.xScaleFactor + header.xOffset);
        p.y = float(y * header.yScaleFactor + header.yOffset);
        p.z = float(z * header.zScaleFactor + header.zOffset);

        points.push_back(p);

        // 跳过其他点属性
        file.seekg(header.pointDataRecordLength - 12, std::ios::cur);
    }

    return points;
}


void bounding(const std::vector<Point>& points, double& minx, double& maxx, double &miny, double &maxy)
{
    minx = std::numeric_limits<double>::max();
    maxx = std::numeric_limits<double>::min();
    miny = std::numeric_limits<double>::max();
    maxy = std::numeric_limits<double>::min();

    for (auto& pt : points)
    {
        if (pt.x > maxx) {
            maxx = pt.x;
        }

        if (pt.x < minx) {
            minx = pt.x;
        }

        if (pt.y < miny) {
            miny = pt.y;
        }

        if (pt.y > maxy) {
            maxy = pt.y;
        }
    }
}

void normalization(std::vector<Point>& points)
{
    double minx = 0
        , maxx = 0
        , miny = 0
        , maxy = 0;

    bounding(points, minx, maxx, miny, maxy);

    for (auto& pt : points)
    {
        pt.x = pt.x - minx;
        pt.y = pt.y - miny;
    }

}


std::vector<size_t> farthestPointSamplingCPU(const std::vector<Point>& points, size_t numSamples) {
    std::vector<size_t> sampledIndices;
    std::vector<double> minDistances(points.size(), std::numeric_limits<double>::max());

    // 随机选择第一个点
    size_t firstIndex = rand() % points.size();
    sampledIndices.push_back(firstIndex);

    for (size_t i = 1; i < numSamples; ++i) {
        size_t farthestIndex = 0;
        double maxMinDistance = 0;

        for (size_t j = 0; j < points.size(); ++j) {
            double dx = points[j].x - points[sampledIndices.back()].x;
            double dy = points[j].y - points[sampledIndices.back()].y;
            double dz = points[j].z - points[sampledIndices.back()].z;
            double distance = dx * dx + dy * dy + dz * dz;

            minDistances[j] = std::min(minDistances[j], distance);

            if (minDistances[j] > maxMinDistance) {
                maxMinDistance = minDistances[j];
                farthestIndex = j;
            }
        }

        sampledIndices.push_back(farthestIndex);
    }

    return sampledIndices;
}


void pcdlist_to_pointlist(const std::vector<PCDData>& pcds, std::vector<Point>& points)
{
    points.resize(pcds.size());
    std::transform(pcds.begin(), pcds.end(), points.begin(), [](const PCDData& d) {
        return d.xyz;
        });
}


int main() {
    //std::vector<Point> points(10240);
    //std::random_device rd;
    //std::mt19937 gen(rd());
    //std::uniform_real_distribution<> dis(-100.0, 100.0);
    //for (auto& p : points) {
    //    p.x = dis(gen);
    //    p.y = dis(gen);
    //    p.z = dis(gen);
    //}

    print_cuda_info();

    std::vector<PCDData> pcds;
    pcds = read_pcd_from_ascii(R"(C:\Users\zhangkh\Desktop\pcdprocess\datasets\labeled\d4\clear\changshaceshiL2-1 - Cloud.txt)");

    std::vector<Point> points;
    pcdlist_to_pointlist(pcds, points);
    normalization(points);
    int numSamples = 4096;

    //std::string filename = R"(E:\DevDatas\pointcloud\changshaceshiL2.las)";
    //readLasfileWithMMP(filename, points);

    // 执行FPS
    std::vector<size_t> sampledIndices = farthestPointSamplingV2(points, numSamples);

    dump_to_ascii("test.txt", pcds, sampledIndices);

    return 0;
}