#include "tityos/ty/ops/bmm.h"

#include <iostream>

namespace ty {
Tensor bmm(const Tensor& tensor1, const Tensor& tensor2) {
    if (tensor1.getDType() != tensor2.getDType()) {
        throw std::invalid_argument(
            "Types must match for batch matrix-matrix product");
    }

    if (tensor1.getNDim() != 3 || tensor2.getNDim() != 3) {
        throw std::invalid_argument("Input tensors must be 3 dimensional");
    }

    auto shape1 = tensor1.getShape();
    auto shape2 = tensor2.getShape();

    if (shape1[0] != shape2[0]) {
        throw std::invalid_argument(
            "Input tensors must contain the same number of matrices");
    }

    if (shape1[2] != shape2[1]) {
        throw std::invalid_argument(
            "Matrices must have compatible dimensions for multiplication");
    }

    auto b = internal::backend::getBackend(tensor1.getDevice().type());
    auto resultBase = std::make_shared<internal::BaseTensor>(
        b->bmm(*tensor1.getBaseTensor(), *tensor2.getBaseTensor()));

    return Tensor(resultBase);
}
} // namespace ty