#pragma once

#include "tityos/ty/tensor/Tensor.h"

namespace ty {
namespace internal {
    BaseTensor internalAddCpu(const BaseTensor& tensor1,
                              const BaseTensor& tensor2);
}
} // namespace ty