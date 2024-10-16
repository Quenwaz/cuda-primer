# CMakeLists.txt
cmake_minimum_required(VERSION 3.26)

enable_language(CUDA)

# 确保启用了 CUDA
find_package(CUDA REQUIRED)

# 添加 CUDA 文件
add_executable(fps fps.cu main.cc)

target_compile_options(fps PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -G
                       -g, -0
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)


message(STATUS "message: CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}")
# 设置 CUDA 计算能力（可选）
set_target_properties(fps PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
