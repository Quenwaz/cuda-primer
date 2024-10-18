# CMakeLists.txt
cmake_minimum_required(VERSION 3.26)

enable_language(CUDA)

# 确保启用了 CUDA
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)


# 添加 CUDA 文件
add_executable(fps fps.cu main.cc)

target_compile_options(fps PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -G
                       -g, -0
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)


                       

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# 设置 CUDA 计算能力（可选）
set_target_properties(fps PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
