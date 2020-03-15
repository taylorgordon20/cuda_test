cmake_minimum_required(VERSION 3.16)

# Add the happy Python extension target.
add_library(happy_ext MODULE happy_ext.cpp)
target_include_directories(happy_ext PUBLIC "${HAPPY_INCLUDE_DIR}")
target_compile_features(happy_ext PUBLIC cxx_std_14)

# Link to dependencies.
target_link_libraries(happy_ext PRIVATE pybind11::module happy_cpu)

# Add GPU implementation if CUDA is available.
if(CMAKE_CUDA_COMPILER)
  target_compile_definitions(happy_ext PRIVATE WITH_CUDA)
  target_link_libraries(happy_ext PRIVATE happy_gpu)
endif()

# Configure the built library to be consumed as a Python extension.
set_target_properties(
  happy_ext
  PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    OUTPUT_NAME "happy"
)