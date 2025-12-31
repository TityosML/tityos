#pragma once

#include "tityos/ty/tensor/Tensor.h"
#include <vector>
#include <algorithm>

namespace ty {
Tensor expand(const Tensor& a, std::vector<size_t> newShape);
}
