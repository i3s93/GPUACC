cmake_minimum_required(VERSION 3.10)

# Name for the project
project(Dgemm_Benchmark)

# Specify which packages are required to run the project
# Note: The Cray compiler wrappers implicitly link Cray LibSci 
# which contains math libraries such as BLAS, LAPACK, ScaLAPACK, BLACS
find_package(CUDA REQUIRED)

# Specify the source files for the code
set(CUDA_SOURCES cublas_dgemm.cu)

# Create the executable target for the CUDA code
add_executable(dgemm_cuda ${CUDA_SOURCES})

# Linut the libraries to the code
target_link_libraries(dgemm_cuda ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})