cmake_minimum_required(VERSION 3.8)

# Add and configure the happy python extension target.
add_library(happy_ext MODULE happy_ext.cpp)
target_link_libraries(happy_ext PRIVATE pybind11::module happy_lib)
target_compile_features(happy_ext PUBLIC cxx_std_14)
set_target_properties(
  happy_ext
  PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    OUTPUT_NAME "happy"
)
