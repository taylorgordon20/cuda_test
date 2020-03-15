#pragma once

#include <vector>

namespace happy {

std::vector<int> add_cpu(
    const std::vector<int>& v1, const std::vector<int>& v2);

std::vector<int> add_gpu(
    const std::vector<int>& v1, const std::vector<int>& v2);

}  // namespace happy