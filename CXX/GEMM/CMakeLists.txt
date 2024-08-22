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

# Set the minimum standard for the languages
set(CMAKE_C_STANDARD    11 CACHE STRING "C language standard")
set(CMAKE_CXX_STANDARD  11 CACHE STRING "C++ language standard")
set(CMAKE_CUDA_STANDARD 11 CACHE STRING "CUDA language standard")

option(CMAKE_C_STANDARD_REQUIRED    "Require the C language standard to set"    ON)
option(CMAKE_CXX_STANDARD_REQUIRED  "Require the C++ language standard to set"  ON)
option(CMAKE_CUDA_STANDARD_REQUIRED "Require the CUDA language standard to set" ON)

option(CMAKE_C_EXTENSIONS    "Enable/disable extensions, e.g. -std=gnu11 vs. -std=c11"     OFF)
option(CMAKE_CXX_EXTENSIONS  "Enable/disable extensions, e.g. -std=gnu++11 vs. -std=c++11" OFF)
option(CMAKE_CUDA_EXTENSIONS "Enable/disable extensions" OFF)


find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Create the executable target for the CUDA code
# The minimum should be C++ 11
add_executable(dgemm_cuda cublas_dgemm.cu)

# Link the libraries to the code
# We also add a directory where these libraries are located
target_link_libraries(dgemm_cuda PRIVATE CUDA::cublas OpenMP::OpenMP_CXX BLAS::BLAS)

add_subdirectory(include/tclap)




