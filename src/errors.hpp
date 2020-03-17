#pragma once

#include <stdexcept>

#define ARGUMENT_CHECK(cond)              \
  do {                                    \
    if (!(cond)) {                        \
      throw std::invalid_argument(#cond); \
    }                                     \
  } while (0)
