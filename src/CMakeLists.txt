cmake_minimum_required(VERSION 3.17)

# Add the hello library CPU target
# TODO: Add pybind11 include directories.
add_library(hello_cpu SHARED hello.cpp)
target_include_directories((hello_cpu PUBLIC ../include)
target_compile_features(hello_cpu PUBLIC cxx_std_14)

# Add the hello library GPU target
add_library(hello_gpu SHARED hello.cu)
target_include_directories((hello_gpu PUBLIC ../include)
target_compile_features(hello_gpu PUBLIC cxx_std_14)
