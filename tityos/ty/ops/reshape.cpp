#include "tityos/ty/ops/reshape.h"

namespace ty {
namespace internal {
    BaseTensor reshape(const BaseTensor& tensor, const TensorShape& newShape,
                       size_t ndim) {
        if (tensor.isContiguous()) {
            return view(tensor, newShape, ndim);
        }

        return view(contiguous(tensor), newShape, ndim);
    }

    BaseTensor view(const BaseTensor& tensor, const TensorShape& newShape,
                    size_t ndim) {
        ShapeStrides newLayout(newShape, ndim);
        return BaseTensor(tensor.getTensorStorage(), newLayout,
                          tensor.getDType());
    }
} // namespace internal

Tensor reshape(const Tensor& tensor, const TensorShape& newShape, size_t ndim) {
    return Tensor(std::make_shared<internal::BaseTensor>(
        internal::reshape(*tensor.getBaseTensor(), newShape, ndim)));
}

Tensor view(const Tensor& tensor, const TensorShape& newShape, size_t ndim) {
    if (!tensor.isContiguous()) {
        throw std::runtime_error("Cannot get view of a non-contiguous tensor");
    }

    return Tensor(std::make_shared<internal::BaseTensor>(
        internal::view(*tensor.getBaseTensor(), newShape, ndim)));
}
} // namespace ty
