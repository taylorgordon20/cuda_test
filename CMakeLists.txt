cmake_minimum_required(VERSION 3.17)

include(CheckLanguage)

project(
  happy
  VERSION 0.0.1
  DESCRIPTION "A dummy project to test building Python C++ and CUDA extensions"
  LANGUAGES CXX
)

if(check_language(CUDA))
  message(STATUS "CUDA support found. Building library with CUDA bindings.")
  enable_language(CUDA)
else()
  message(STATUS "No CUDA support found. Building library without CUDA bindings.")
endif()


add_subdirectory(src)