#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "happy.hpp"

PYBIND11_MODULE(happy, m) {
  m.doc() = "Dummy module for adding vectors";
  m.def("add_cpu", &happy::add_cpu, "A function which adds two vectors on CPU");
  m.def("add_gpu", &happy::add_gpu, "A function which adds two vectors on GPU");
}