cmake_minimum_required(VERSION 3.16)

include(CheckLanguage)

project(
  happy
  VERSION 0.0.1
  DESCRIPTION "A dummy project to test building Python C++ and CUDA extensions"
  LANGUAGES CXX
)

#== BUILD OPTIONS ==#

# User options controlling what we build.
option(HAPPY_ENABLE_PYTHON "Whether to build the happy CPython extension" ON)

#== BUILD VARIABLES ==#

# Path to the public include directory.
set(HAPPY_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")

# Check whether CUDA is supported and if so enable the appropriate build settings.
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA support found. Building library with CUDA bindings.")
  enable_language(CUDA)
else()
  message(STATUS "No CUDA support found. Building library without CUDA bindings.")
endif()

#== BUILD TARGETS ==#

# Add third-party dependencies.
add_subdirectory(third_party)

# Add our source targets.
add_subdirectory(src)

# Add python extension targets.
if(HAPPY_ENABLE_PYTHON)
  add_subdirectory(python)
endif()

#== BUILD TESTS ==#
