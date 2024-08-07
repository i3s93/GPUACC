cmake_minimum_required(VERSION 3.18)

# This has to be set before project()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Name for the project
# Note that for CUDA codes, it is better to specify
# CUDA as a language than use "find_package(CUDA REQUIRED)"
# as the latter is deprecated.
project(Dgemm_Benchmark LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Create the executable target for the CUDA code
# The minimum should be C++ 11
add_executable(dgemm_cuda cublas_dgemm.cu)
target_compile_features(dgemm_cuda PRIVATE cxx_std_11)

# Link the libraries to the code
# We also add a directory where these libraries are located
target_link_libraries(dgemm_cuda PRIVATE CUDA::cublas OpenMP::OpenMP_CXX)






