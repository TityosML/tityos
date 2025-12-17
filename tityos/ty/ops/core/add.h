#pragma once

#include "tityos/ty/ops/cpu/add.h"
#include "tityos/ty/ops/cpu/avx2/add.h"
#include "tityos/ty/ops/defines.h"
#include "tityos/ty/tensor/Tensor.h"

namespace ty {
namespace internal {
    DECLARE_NO_CUDA_DISPATCH_FUNCTION(internalAdd)
    DECLARE_AVX2_DISPATCH_FUNCTION(internalAdd)
    DEFINE_FUNC_DISPATCH(internalAdd)
} // namespace internal
} // namespace ty