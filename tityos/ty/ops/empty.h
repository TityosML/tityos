#pragma once

#include "tityos/ty/tensor/BaseTensor.h"

namespace ty {
namespace internal {
    BaseTensor empty(TensorShape shape, size_t ndim, DType dtype, Device device);

    BaseTensor emptyLike(BaseTensor tensor);
}
} // namespace ty