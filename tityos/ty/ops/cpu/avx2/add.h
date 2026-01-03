#pragma once

#include "tityos/ty/tensor/Tensor.h"

namespace ty {
namespace internal {
    BaseTensor internalAddAvx2(const BaseTensor& tensor1,
                               const BaseTensor& tensor2);
}
} // namespace ty