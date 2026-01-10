#include "tityos/ty/ops/expand.h"

namespace ty {
Tensor expand(const Tensor& tensor, std::vector<size_t> newShape) {
    size_t newNDim = newShape.size();
    size_t oldNDim = tensor.getNDim();
    internal::TensorStrides newStrides;
    internal::TensorShape newShapeArray;
    auto baseTensor = tensor.getBaseTensor();
    auto oldShape = baseTensor->getShape();
    auto oldStrides = baseTensor->getStrides();
    size_t shapeOffset = oldNDim - newNDim;

    if (newNDim < oldNDim) {
        for (size_t i = 0; i < shapeOffset; i++) {
            if (oldShape[i] != 1) {
                throw std::invalid_argument(
                    "Cannot remove non-singletion dimensions during expansion");
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

    auto offset = tensor.getBaseTensor()->getLayout().getOffset();
    internal::ShapeStrides newLayout(newShapeArray, newStrides, offset,
                                     newNDim);
    auto newBaseTensor = std::make_shared<internal::BaseTensor>(
        baseTensor->getTensorStorage(), newLayout, tensor.getDType());

    return Tensor(newBaseTensor);
}
} // namespace ty