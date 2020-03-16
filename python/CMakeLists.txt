cmake_minimum_required(VERSION 3.16)

# Add the happy Python extension target.
add_library(happy_ext MODULE happy_ext.cpp)
target_compile_features(happy_ext PUBLIC cxx_std_14)
target_compile_definitions(happy_ext PRIVATE $<$<BOOL:CMAKE_CUDA_COMPILER>:WITH_CUDA>)

# Link to dependencies.
target_link_libraries(happy_ext PRIVATE pybind11::module happy_lib)

# Configure the built library to be consumed as a Python extension.
set_target_properties(
  happy_ext
  PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    OUTPUT_NAME "happy"
)