# CMakeLists.txt
cmake_minimum_required(VERSION 3.26)


# 确保启用了 CUDA
find_package(CUDA REQUIRED)

# 添加 CUDA 文件
add_executable(cuda_project main.cu)

# 设置 CUDA 计算能力（可选）
set_target_properties(cuda_project PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
