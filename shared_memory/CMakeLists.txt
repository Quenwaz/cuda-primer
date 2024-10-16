# CMakeLists.txt
cmake_minimum_required(VERSION 3.26)


# 确保启用了 CUDA
find_package(CUDA REQUIRED)

# 添加 CUDA 文件
add_executable(shared_memory shared_memory.cu)


target_compile_options(shared_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -G
                       -g, -0
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

# 设置 CUDA 计算能力（可选）
set_target_properties(shared_memory PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
