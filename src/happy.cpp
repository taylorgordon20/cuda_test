#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "errors.hpp"
#include "happy.hpp"

namespace happy {

std::vector<int> add_cpu(
    const std::vector<int>& v1, const std::vector<int>& v2) {
  ARGUMENT_CHECK(v1.size() == v2.size());

  std::vector<int> ret;
  ret.reserve(v1.size());
  for (int i = 0; i < v1.size(); i += 1) {
    ret.push_back(v1[i] + v2[i]);
  }
  return ret;
}

}  // namespace happy