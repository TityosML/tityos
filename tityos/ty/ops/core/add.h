#pragma once

#include "tityos/ty/ops/cpu/add.h"
#include "tityos/ty/ops/cpu/avx2/add.h"
#include "tityos/ty/ops/cuda/add.h"
#include "tityos/ty/ops/defines.h"
#include "tityos/ty/tensor/Tensor.h"

namespace ty {
namespace internal {
    DECLARE_CUDA_DISPATCH_FUNCTION(internalAdd)
    DECLARE_AVX2_DISPATCH_FUNCTION(internalAdd)
    DEFINE_FUNC_DISPATCH(internalAdd, BaseTensor)
} // namespace internal
} // namespace ty