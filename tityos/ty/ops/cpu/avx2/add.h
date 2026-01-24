#pragma once

#include "tityos/ty/ops/cpu/avx2/avx2Traits.h"
#include "tityos/ty/ops/cpu/TensorView.h"
#include "tityos/ty/tensor/Tensor.h"

#include <immintrin.h>

namespace ty {
namespace internal {
    void addAvx2(BaseTensor& result, const BaseTensor& tensor1,
                       const BaseTensor& tensor2);
}
} // namespace ty