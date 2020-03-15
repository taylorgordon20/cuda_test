cmake_minimum_required(VERSION 3.16)

include(CheckLanguage)

project(
  happy
  VERSION 0.0.1
  DESCRIPTION "A dummy project to test building Python C++ and CUDA extensions"
  LANGUAGES CXX
)

# Flags controlling what we build.
set(BUILD_PYTHON_EXTENSION ON CACHE BOOL "Whether to build the happy CPython extension")

# Check whether CUDA is supported and if so enable the appropriate build declarations.
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA support found. Building library with CUDA bindings.")
  enable_language(CUDA)
else()
  message(STATUS "No CUDA support found. Building library without CUDA bindings.")
endif()

# Add third-party dependencies.
add_subdirectory(third_party)

# Add our source targets.
add_subdirectory(src)

# Add python extension targets.
add_subdirectory(py)
install(TARGETS happy_ext LIBRARY DESTINATION happy)