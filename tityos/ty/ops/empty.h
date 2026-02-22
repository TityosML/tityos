#pragma once

#include "tityos/ty/export.h"
#include "tityos/ty/tensor/Tensor.h"

namespace ty {
namespace internal {
    /*
    TODO: Figure out why these need to be exported
    */
    BaseTensor TITYOS_API empty(const TensorShape& shape, size_t ndim, DType dtype, Device device);

    BaseTensor TITYOS_API emptyLike(const BaseTensor& tensor);
} // namespace internal

Tensor TITYOS_API empty(const TensorShape& shape, size_t ndim, DType dtype, Device device);

Tensor TITYOS_API emptyLike(const Tensor& tensor);
} // namespace ty