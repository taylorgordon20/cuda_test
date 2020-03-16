cmake_minimum_required(VERSION 3.16)

# Add the hello library CPU target
add_library(happy_cpu_lib STATIC happy_cpu.cpp errors.hpp)
target_include_directories(happy_cpu_lib PUBLIC "${HAPPY_INCLUDE_DIR}")
target_compile_features(happy_cpu_lib PUBLIC cxx_std_14)

# Add the happy library GPU target
if(CMAKE_CUDA_COMPILER)
  add_library(happy_gpu_lib STATIC happy_gpu.cu errors.hpp)
  target_include_directories(happy_gpu_lib PUBLIC "${HAPPY_INCLUDE_DIR}")
  target_compile_features(happy_gpu_lib PUBLIC cxx_std_14)
endif()

# For convience, build a library with both CPU and GPU targets.
# TODO: Use generator syntax to optionally link to GPU target.
add_library(happy_lib INTERFACE)
target_link_libraries(happy_lib INTERFACE happy_cpu_lib)
if(CMAKE_CUDA_COMPILER)
  target_link_libraries(happy_lib INTERFACE happy_gpu_lib)
endif()