#pragma once

#include "tityos/ty/tensor/Tensor.h"

#include <cstdint>
#include <stdexcept>

namespace ty {
namespace internal {
    BaseTensor internalAddCpu(const BaseTensor& tensor1,
                              const BaseTensor& tensor2);
}
} // namespace ty