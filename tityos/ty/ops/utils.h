#pragma once

#include "tityos/ty/tensor/BaseTensor.h"

#include <array>
#include <cstdlib>
#include <vector>

namespace ty {
namespace internal {
    std::vector<size_t> broadcastShape(
        const std::array<size_t, MAX_DIMS>& tensor1Shape, size_t tensor1NDim,
        const std::array<size_t, MAX_DIMS>& tensor2Shape, size_t tensor2NDim);
}
} // namespace ty