cmake_minimum_required(VERSION 3.10)
project(ray_tracer LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)

enable_language(CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include_directories(
    kernels
    utility
)

add_executable(
    ray_tracer
    main.cu
    lodepng.cpp
)