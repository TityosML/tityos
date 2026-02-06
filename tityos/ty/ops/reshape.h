#pragma once

#include "tityos/ty/tensor/Tensor.h"
#include "tityos/ty/ops/contiguous.h"

namespace ty {
namespace internal {
    BaseTensor reshape(const BaseTensor& tensor, const TensorShape& newShape,
                       size_t ndim);
    BaseTensor view(const BaseTensor& tensor, const TensorShape& newShape,
                    size_t ndim);
} // namespace internal

Tensor reshape(const Tensor& tensor, const TensorShape& newShape, size_t ndim);
Tensor view(const Tensor& tensor, const TensorShape& newShape, size_t ndim);
} // namespace ty
