cmake_minimum_required(VERSION 3.16)

# Add the hello library CPU target
add_library(happy_cpu STATIC happy.cpp errors.hpp)
target_include_directories(happy_cpu PUBLIC ../include)
target_compile_features(happy_cpu PUBLIC cxx_std_14)

# Add the happy library GPU target
add_library(happy_gpu STATIC happy.cu errors.hpp)
target_include_directories(happy_gpu PUBLIC ../include)
target_compile_features(happy_gpu PUBLIC cxx_std_14)