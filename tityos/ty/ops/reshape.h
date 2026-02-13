#pragma once

#include "tityos/ty/export.h"
#include "tityos/ty/ops/contiguous.h"
#include "tityos/ty/tensor/Tensor.h"

namespace ty {
namespace internal {
    BaseTensor reshape(const BaseTensor& tensor, const TensorShape& newShape,
                       size_t ndim);
    BaseTensor view(const BaseTensor& tensor, const TensorShape& newShape,
                    size_t ndim);
} // namespace internal

Tensor TITYOS_API reshape(const Tensor& tensor, const TensorShape& newShape,
                          size_t ndim);
Tensor TITYOS_API view(const Tensor& tensor, const TensorShape& newShape,
                       size_t ndim);
} // namespace ty
