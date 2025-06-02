cmake_minimum_required(VERSION 3.10)

project(matrix_multiplication LANGUAGES CXX CUDA)

# CPU
set(SOURCES_CPU
    src/matrix_cpu.cpp
    src/matrix.cpp
    src/cpu_main.cpp
)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(cpu_matrix_multiplication ${SOURCES_CPU})

target_include_directories(cpu_matrix_multiplication
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(cpu_matrix_multiplication
    PRIVATE
        OpenMP::OpenMP_CXX
)

target_compile_options(cpu_matrix_multiplication PRIVATE -std=c++11 -fopenmp)

# GPU
set(SOURCES_GPU
    src/matrix_gpu.cu
    src/matrix.cpp
    src/gpu_main.cu
)

add_executable(gpu_matrix_multiplication ${SOURCES_GPU})

target_include_directories(gpu_matrix_multiplication
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(gpu_matrix_multiplication
    PRIVATE
        cublas
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
# only use when debug: -O0 -g -G
set_target_properties(gpu_matrix_multiplication PROPERTIES CUDA_ARCHITECTURES 89)
set_target_properties(gpu_matrix_multiplication PROPERTIES CUDA_SEPARABLE_COMPILATION ON)