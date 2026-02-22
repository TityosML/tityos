#include "tityos/ty/ops/empty.h"

namespace ty {
namespace internal {
    BaseTensor empty(const TensorShape& shape, size_t ndim, DType dtype, Device device) {
        auto numBytes =
            std::accumulate(shape.begin(), shape.begin() + ndim, 1, std::multiplies<size_t>()) * dtypeSize(dtype);
        auto resultStorage = std::make_shared<TensorStorage>(numBytes, device);
        ShapeStrides resultLayout(shape, ndim);
        return BaseTensor(resultStorage, resultLayout, dtype);
    }

    BaseTensor emptyLike(const BaseTensor& tensor) {
        auto resultStorage = std::make_shared<TensorStorage>(tensor.getLogicalSize(), tensor.getDevice());
        ShapeStrides resultLayout(tensor.getShape(), tensor.getNDim());
        return BaseTensor(resultStorage, resultLayout, tensor.getDType());
    }
} // namespace internal

Tensor empty(const TensorShape& shape, size_t ndim, DType dtype, Device device) {
    return Tensor(std::make_shared<internal::BaseTensor>(internal::empty(shape, ndim, dtype, device)));
}

Tensor emptyLike(const Tensor& tensor) {
    return Tensor(std::make_shared<internal::BaseTensor>(internal::emptyLike(*tensor.getBaseTensor())));
}
} // namespace ty