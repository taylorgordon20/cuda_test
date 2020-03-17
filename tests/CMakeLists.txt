cmake_minimum_required(VERSION 3.8)

# Add and configure the happy test executable target.
add_executable(happy_test_exe happy_test.cpp)
target_link_libraries(happy_test_exe PRIVATE Catch2::Catch2 happy_lib)
target_compile_features(happy_test_exe PRIVATE cxx_std_14)
target_compile_definitions(happy_ext PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:WITH_CUDA>)
set_target_properties(happy_test_exe PROPERTIES OUTPUT_NAME "happy_test")

# Add a ctest target to run the happy test executable.
add_test(NAME happy_test COMMAND happy_test_exe)
