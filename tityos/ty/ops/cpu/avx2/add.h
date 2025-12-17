#pragma once

#include "tityos/ty/tensor/Tensor.h"

namespace ty {
namespace internal {
    void internalAddAvx2(Tensor& result, const Tensor& tensor1,
                         const Tensor& tensor2);
}
} // namespace ty