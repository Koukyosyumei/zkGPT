cmake_minimum_required(VERSION 3.10)
project(zkCNN)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3 -pthread")
link_directories(3rd/mcl)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found — parallel proving enabled")
else()
    message(STATUS "OpenMP not found — falling back to single-threaded proving")
    # Stub target: #pragma omp directives are silently ignored without -fopenmp.
    # omp.h stubs are pulled in via the GCC include path below.
    add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
        INTERFACE_COMPILE_OPTIONS "-I/usr/lib/gcc/aarch64-linux-gnu/11/include"
        INTERFACE_LINK_LIBRARIES "pthread")
endif()

include_directories(src)
include_directories(3rd)
include_directories(3rd/mcl/include)

add_subdirectory(src)
add_subdirectory(3rd/mcl)