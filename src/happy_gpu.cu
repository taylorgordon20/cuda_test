#include <limits>
#include <vector>

#include "errors.hpp"

namespace happy {

__global__ void add_kernel(int* a, int* b, int* c, size_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

std::vector<int> add_gpu(
    const std::vector<int>& v1, const std::vector<int>& v2) {
  ARGUMENT_CHECK(v1.size() == v2.size());
  ARGUMENT_CHECK(v1.size() <= std::numeric_limits<unsigned int>::max());

  int* d_a = nullptr;
  int* d_b = nullptr;
  int* d_c = nullptr;

  auto n = v1.size();
  auto alloc_size = sizeof(int) * n;
  std::vector<int> ret(n);

  cudaMalloc(&d_a, alloc_size);
  cudaMalloc(&d_b, alloc_size);
  cudaMalloc(&d_c, alloc_size);

  cudaMemcpy(d_a, v1.data(), alloc_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, v2.data(), alloc_size, cudaMemcpyHostToDevice);

  auto kBlockSize = 256u;
  auto grid_size = static_cast<unsigned int>(1 + (n - 1) / kBlockSize);
  add_kernel<<<grid_size, kBlockSize>>>(d_a, d_b, d_c, n);

  cudaMemcpy(ret.data(), d_c, alloc_size, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return ret;
}

}  // namespace happy
