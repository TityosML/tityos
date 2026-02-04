#include "tityos/ty/ops/empty.h"

namespace ty {
namespace internal {
    BaseTensor empty(TensorShape shape, size_t ndim, DType dtype, Device device) {
        auto numBytes = std::accumulate(shape.begin(), shape.begin() + ndim, 1,
                                        std::multiplies<size_t>()) *
                        dtypeSize(dtype);
        auto resultStorage = std::make_shared<TensorStorage>(numBytes, device);
        ShapeStrides resultLayout(shape, ndim);
        return BaseTensor(resultStorage, resultLayout, dtype);
    }

    BaseTensor emptyLike(BaseTensor tensor) {
        auto resultStorage = std::make_shared<TensorStorage>(
            tensor.getLogicalSize(), tensor.getDevice());
        ShapeStrides resultLayout(tensor.getShape(), tensor.getNDim());
        return BaseTensor(resultStorage, resultLayout, tensor.getDType());
    }
} // namespace internal
} // namespace ty