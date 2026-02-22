#pragma once

#include "tityos/ty/export.h"
#include "tityos/ty/tensor/Tensor.h"

#include <algorithm>
#include <vector>

namespace ty {
namespace internal {
    BaseTensor expand(const BaseTensor& tensor, const std::vector<size_t>& newShape);
}
Tensor TITYOS_API expand(const Tensor& tensor, const std::vector<size_t>& newShape);
} // namespace ty
