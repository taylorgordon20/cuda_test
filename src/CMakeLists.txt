cmake_minimum_required(VERSION 3.8)

# List all implementation source files and header files.
file(GLOB_RECURSE HAPPY_IMPL_HDRS CONFIGURE_DEPENDS "*.h" "*.hpp")
set(HAPPY_IMPL_SRCS happy_cpu.cpp)

# Add CUDA implementation source files if language supported.
if(CMAKE_CUDA_COMPILER)
  list(APPEND HAPPY_IMPL_SRCS happy_gpu.cu)
endif()

# Add and configure the hello library target.
add_library(happy_lib STATIC ${HAPPY_IMPL_HDRS} ${HAPPY_IMPL_SRCS})
target_include_directories(happy_lib PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(happy_lib PUBLIC cxx_std_14)
set_target_properties(happy_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
