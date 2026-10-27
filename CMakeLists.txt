cmake_minimum_required(VERSION 3.18)

project(matrix_multiplication LANGUAGES CXX)
option(BUILD_CUDA "Build the CUDA benchmarks (requires NVIDIA CUDA Toolkit)" ON)

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

target_compile_features(cpu_matrix_multiplication PRIVATE cxx_std_17)

# GPU
if(BUILD_CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
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
        CUDA::cublas
)

target_compile_features(gpu_matrix_multiplication PRIVATE cxx_std_17)
set_target_properties(gpu_matrix_multiplication PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
# Pass -DCMAKE_CUDA_ARCHITECTURES=<compute capability without dot> to override.
set_target_properties(gpu_matrix_multiplication PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
