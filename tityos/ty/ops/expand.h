#pragma once

#include "tityos/ty/tensor/Tensor.h"

#include <algorithm>
#include <vector>

namespace ty {
namespace internal {
    BaseTensor expand(const Tensor& tensor, std::vector<size_t> newShape);
}
Tensor expand(const Tensor& tensor, std::vector<size_t> newShape);
} // namespace ty
