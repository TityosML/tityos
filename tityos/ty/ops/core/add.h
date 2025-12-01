#pragma once

#include "tityos/ty/ops/cpu/add.h"
#include "tityos/ty/ops/defines.h"
#include "tityos/ty/tensor/Tensor.h"

namespace ty {
namespace internal {
    DEFINE_FUNC_DISPATCH(internalAdd)
}
} // namespace ty