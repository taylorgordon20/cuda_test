cmake_minimum_required(VERSION 3.16)

# Add the unit tests target.
add_executable(happy_test_exe happy_test.cpp)
target_compile_features(happy_test_exe PRIVATE cxx_std_14)
target_compile_definitions(happy_test_exe PRIVATE $<$<BOOL:CMAKE_CUDA_COMPILER>:WITH_CUDA>)
set_target_properties(happy_test_exe PROPERTIES OUTPUT_NAME "happy_test")

# Link to dependencies.
target_link_libraries(happy_test_exe PRIVATE Catch2::Catch2 happy_lib)

# Add a ctest target.
add_test(NAME happy_test COMMAND happy_test_exe)