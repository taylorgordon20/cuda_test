#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <happy/happy.hpp>

TEST_CASE("Tests calling the add_cpu function", "[add_cpu]") {
  std::vector<int> a = {1, 2, 3, 4, 5};
  std::vector<int> b = {9, 7, 5, 3, 1};

  auto c = happy::add_cpu(a, b);
  REQUIRE_THAT(c, Catch::Equals<int>({10, 9, 8, 7, 6}));
}

#ifdef WITH_CUDA
TEST_CASE("Tests calling the add_gpu function", "[add_cpu]") {
  std::vector<int> a = {1, 2, 3, 4, 5};
  std::vector<int> b = {9, 7, 5, 3, 1};

  auto c = happy::add_gpu(a, b);
  REQUIRE_THAT(c, Catch::Equals<int>({10, 9, 8, 7, 6}));
}
#endif
