cmake_minimum_required(VERSION 3.16)

# Add the hello library CPU target
add_library(happy_cpu STATIC happy_cpu.cpp errors.hpp)
target_compile_features(happy_cpu PUBLIC cxx_std_14)

# Add the happy library GPU target
add_library(happy_gpu STATIC happy_gpu.cu errors.hpp)
target_compile_features(happy_gpu PUBLIC cxx_std_14)