#include "tityos/ty/ops/bmm.h"

namespace ty {
namespace internal {
    BaseTensor matmul(const BaseTensor& tensor1, const BaseTensor& tensor2) {
        return bmm(tensor1, tensor2);
    }

    BaseTensor bmm(const BaseTensor& tensor1, const BaseTensor& tensor2) {
        auto b = internal::backend::getBackend(tensor1.getDevice().type());
        return b->bmm(tensor1, tensor2);
    }

} // namespace internal

Tensor matmul(const Tensor& tensor1, const Tensor& tensor2) {
    if (tensor1.getDType() != tensor2.getDType()) {
        throw std::invalid_argument("Types must match for batch matrix-matrix product");
    }

    auto shape1 = tensor1.getShape();
    auto shape2 = tensor2.getShape();

    if (tensor1.getNDim() < 2 || tensor2.getNDim() < 2) {
        throw std::invalid_argument("Cannot perform matrix multiplication on "
                                    "tensors with less than 2 dimensions");
    }

    size_t N1 = shape1[tensor1.getNDim() - 1];
    size_t N2 = shape2[tensor1.getNDim() - 2];

    if (N1 != N2) {
        throw std::invalid_argument("Matrices must have compatible dimensions for multiplication");
    }

    return Tensor(
        std::make_shared<internal::BaseTensor>(internal::matmul(*tensor1.getBaseTensor(), *tensor2.getBaseTensor())));
}

Tensor bmm(const Tensor& tensor1, const Tensor& tensor2) {
    if (tensor1.getDType() != tensor2.getDType()) {
        throw std::invalid_argument("Types must match for batch matrix-matrix product");
    }

    if (tensor1.getNDim() != 3 || tensor2.getNDim() != 3) {
        throw std::invalid_argument("Input tensors must be 3 dimensional");
    }

    auto shape1 = tensor1.getShape();
    auto shape2 = tensor2.getShape();

    if (shape1[0] != shape2[0]) {
        throw std::invalid_argument("Input tensors must contain the same number of matrices");
    }

    if (shape1[2] != shape2[1]) {
        throw std::invalid_argument("Matrices must have compatible dimensions for multiplication");
    }

    return Tensor(
        std::make_shared<internal::BaseTensor>(internal::bmm(*tensor1.getBaseTensor(), *tensor2.getBaseTensor())));
}
} // namespace ty