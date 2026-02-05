#pragma once

#include "tityos/ty/ops/cpu/TensorView.h"
#include "tityos/ty/ops/cpu/avx/avxTraits.h"
#include "tityos/ty/ops/dispatchDType.h"
#include "tityos/ty/tensor/Tensor.h"

#include <immintrin.h>
#include <omp.h>

namespace ty {
namespace internal {
    void addAvx(BaseTensor& result, const BaseTensor& tensor1,
                const BaseTensor& tensor2);
}
} // namespace ty