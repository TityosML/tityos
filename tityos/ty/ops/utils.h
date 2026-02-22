#pragma once

#include "tityos/ty/tensor/BaseTensor.h"

#include <array>
#include <cstdlib>
#include <vector>

namespace ty {
namespace internal {
    std::vector<size_t> broadcastShape(const TensorShape& tensor1Shape, size_t tensor1NDim,
                                       const TensorShape& tensor2Shape, size_t tensor2NDim);
}
} // namespace ty