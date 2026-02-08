#pragma once

#include "tityos/ty/tensor/Tensor.h"

namespace ty {
namespace internal {
    BaseTensor empty(const TensorShape& shape, size_t ndim, DType dtype,
                     Device device);

    BaseTensor emptyLike(const BaseTensor& tensor);
} // namespace internal

Tensor empty(const TensorShape& shape, size_t ndim, DType dtype, Device device);

Tensor emptyLike(const Tensor& tensor);
} // namespace ty