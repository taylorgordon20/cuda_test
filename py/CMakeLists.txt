cmake_minimum_required(VERSION 3.16)

# Add the hello CPython extension target.
pybind11_add_module(happy_ext happy_ext.cpp)
target_include_directories(happy_ext PUBLIC ../include)
target_link_libraries(happy_ext PRIVATE happy_cpu)

# If CUDA is available, expose and link to the GPU implementation.
if(CMAKE_CUDA_COMPILER)
  target_link_libraries(happy_ext PRIVATE happy_gpu)
  target_compile_definitions(happy_ext PRIVATE WITH_CUDA)
endif()