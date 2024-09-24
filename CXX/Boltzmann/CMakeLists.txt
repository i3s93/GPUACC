cmake_minimum_required(VERSION 3.18)

# Name for the project
# Note that for CUDA codes, it is better to specify
# CUDA as a language than use "find_package(CUDA REQUIRED)"
# as the latter is deprecated.
project(Boltzmann_Benchmark LANGUAGES C CXX)

set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "None")
endif()

find_package(OpenMP REQUIRED)
#find_package(CUDAToolkit REQUIRED)

# This has to be set before project()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

#----------------------------------------------------------------------------#
# Options
#----------------------------------------------------------------------------#
option(USE_AVX512 "Enable AVX-512 architecture flags" OFF)
option(USE_SANTITIZER "Enable leak detection" OFF)

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

set(CMAKE_C_FLAGS    "-O3 -ffast-math -fopenmp")
set(CMAKE_CXX_FLAGS  "-O3 -faligned-new -ffast-math -fopenmp")

# Create the executable targets for the different codes
add_executable(maxwell_bkw_serial_ex maxwell_bkw_serial.cpp)
add_executable(maxwell_bkw_omp_ex maxwell_bkw_omp.cpp)

# Create interface target with compiler flags
add_library(Boltzmann)
add_subdirectory(Utilities)
add_subdirectory(SphericalDesign)
add_subdirectory(Collisions)

# Specify the location of the header files for the argument parser
target_include_directories(maxwell_bkw_serial_ex PRIVATE Include)
target_include_directories(maxwell_bkw_omp_ex PRIVATE Include)

# Link the libraries to the code
# Cray compilers automatically provide the link commands to fftw3 and other math libraries
target_link_libraries(maxwell_bkw_serial_ex PRIVATE Boltzmann OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
target_link_libraries(maxwell_bkw_omp_ex PRIVATE Boltzmann OpenMP::OpenMP_C OpenMP::OpenMP_CXX)


