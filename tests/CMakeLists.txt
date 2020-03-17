cmake_minimum_required(VERSION 3.8)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Add the unit tests target.
add_executable(happy_test_exe happy_test.cpp)
set_target_properties(happy_test_exe PROPERTIES OUTPUT_NAME "happy_test")
target_compile_features(happy_test_exe PRIVATE cxx_std_14)
if(CMAKE_CUDA_COMPILER)
  target_compile_definitions(happy_test_exe PRIVATE WITH_CUDA)
endif()

# Link to dependencies.
target_link_libraries(happy_test_exe PRIVATE Catch2::Catch2 happy_lib)

# Add a ctest target.
add_test(NAME happy_test COMMAND happy_test_exe)
