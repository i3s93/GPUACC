cmake_minimum_required(VERSION 3.18)

# This has to be set before project()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Name for the project
project(Dgemm_Benchmark LANGUAGES CXX CUDA)

# Create the executable target for the CUDA code
# The minimum should be C++ 11
add_executable(dgemm_cuda cublas_dgemm.cu)
target_compile_features(dgemm_cuda PRIVATE cxx_std_11)

# Link the libraries to the code
target_link_libraries(dgemm_cuda PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
