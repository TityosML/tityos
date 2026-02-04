#include "tityos/ty/ops/expand.h"

namespace ty {
namespace internal {
    Tensor expand(const BaseTensor& tensor, std::vector<size_t> newShape) {
        size_t newNDim = newShape.size();
        size_t oldNDim = tensor.getNDim();
        TensorStrides newStrides;
        TensorShape newShapeArray;
        auto oldShape = tensor.getShape();
        auto oldStrides = tensor.getStrides();
        size_t shapeOffset = oldNDim - newNDim;

        if (newNDim < oldNDim) {
            for (size_t i = 0; i < shapeOffset; i++) {
                if (oldShape[i] != 1) {
                    throw std::invalid_argument("Cannot remove non-singletion "
                                                "dimensions during expansion");
                }
            }
        } else {
            for (size_t i = 0; i < abs(shapeOffset); i++) {
                newStrides[i] = 0;
                newShapeArray[i] = newShape[i];
            }
        }

        size_t startDim = std::max(0, abs(shapeOffset));
        for (size_t i = startDim; i < newNDim; i++) {
            if (newShape[i] == oldShape[i + shapeOffset]) {
                newStrides[i] = oldStrides[i + shapeOffset];
            } else {
                if (oldShape[i + shapeOffset] != 1) {
                    throw std::invalid_argument(
                        "Cannot expand non-singletion dimension to new size");
                }
                newStrides[i] = 0;
            }

            newShapeArray[i] = newShape[i];
        }

        auto offset = tensor.getLayout().getOffset();
        internal::ShapeStrides newLayout(newShapeArray, newStrides, offset,
                                         newNDim);
        auto newBaseTensor = std::make_shared<internal::BaseTensor>(
            tensor.getTensorStorage(), newLayout, tensor.getDType());

        return Tensor(newBaseTensor);
    }
} // namespace internal

Tensor expand(const Tensor& tensor, std::vector<size_t> newShape) {
    return Tensor(internal::expand(*tensor.getBaseTensor(), newShape));
}
} // namespace ty