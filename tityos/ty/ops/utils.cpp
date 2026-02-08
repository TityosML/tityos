#include "tityos/ty/ops/utils.h"

namespace ty {
namespace internal {
    std::vector<size_t> broadcastShape(const TensorShape& tensor1Shape,
                                       size_t tensor1NDim,
                                       const TensorShape& tensor2Shape,
                                       size_t tensor2NDim) {
        auto largerShape = tensor1Shape;
        auto smallerShape = tensor2Shape;
        size_t broadcastNDim = std::max(tensor1NDim, tensor2NDim);
        std::vector<size_t> broadcastShape;
        broadcastShape.resize(broadcastNDim);

        size_t shapeOffset = abs(tensor1NDim - tensor2NDim);

        if (tensor2NDim > tensor1NDim) {
            largerShape = tensor2Shape;
            smallerShape = tensor1Shape;
        }

        for (size_t i = 0; i < shapeOffset; i++) {
            broadcastShape[i] = largerShape[i];
        }

        for (size_t i = shapeOffset; i < broadcastNDim; i++) {
            if (smallerShape[i - shapeOffset] != largerShape[i]) {
                if (smallerShape[i - shapeOffset] != 1 && largerShape[i] != 1) {
                    throw std::invalid_argument(
                        "Mismatch at non-singleton dimension, cannot broadcast "
                        "shapes");
                }
            }

            broadcastShape[i] =
                std::max(smallerShape[i - shapeOffset], largerShape[i]);
        }

        return broadcastShape;
    }
} // namespace internal
} // namespace ty