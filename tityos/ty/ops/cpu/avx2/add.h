#pragma once

#include "tityos/ty/tensor/Tensor.h"

#include <iostream>

namespace ty {
namespace internal {
    void internalAddAvx2(Tensor& result, const Tensor& tensor1, const Tensor& tensor2);
}
} // namespace ty