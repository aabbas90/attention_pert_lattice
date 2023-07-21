project(ATTENTION_PERT_LATTICE LANGUAGES CXX CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

set(CMAKE_CUDA_ARCHITECTURES native)
# set_property(TARGET tgt PROPERTY CUDA_ARCHITECTURES native)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda -fmax-errors=1000")

include_directories(include)

include(external/CPM.cmake/cmake/CPM.cmake)
#CPMAddPackage(
#  NAME cuco
#  GITHUB_REPOSITORY NVIDIA/cuCollections
#  GIT_TAG dev
#  OPTIONS
#     "BUILD_TESTS OFF"
#     "BUILD_BENCHMARKS OFF"
#     "BUILD_EXAMPLES OFF"
#)

add_library(ATTENTION_PERT_LATTICE INTERFACE)

target_include_directories(ATTENTION_PERT_LATTICE INTERFACE include/)
target_compile_features(ATTENTION_PERT_LATTICE INTERFACE cxx_std_17)
target_compile_options(ATTENTION_PERT_LATTICE INTERFACE -march=native)
target_compile_options(ATTENTION_PERT_LATTICE INTERFACE -fPIC)
target_include_directories(ATTENTION_PERT_LATTICE INTERFACE external/Eigen)
# target_link_libraries(ATTENTION_PERT_LATTICE INTERFACE cuco)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_CPP_STANDARD -std=c++17)
set(PYBIND11_INSTALL ON CACHE BOOL "enable pybind11 bindings.")
add_subdirectory(external/pybind11)

add_subdirectory(src)
add_subdirectory(test)
