#include "tityos/ty/ops/add.h"

#include <iostream>

namespace ty {
Tensor add(const Tensor& tensor1, const Tensor& tensor2) {
    auto broadcastShape =
        internal::broadcastShape(tensor1.getShape(), tensor1.getNDim(),
                                 tensor2.getShape(), tensor2.getNDim());

    auto broadcastedTensor1 = expand(tensor1, broadcastShape);
    auto broadcastedTensor2 = expand(tensor2, broadcastShape);

    auto b = internal::backend::getBackend(tensor1.getDevice().type());
    auto resultBase = std::make_shared<internal::BaseTensor>(
        b->add(*broadcastedTensor1.getBaseTensor(),
               *broadcastedTensor2.getBaseTensor()));

    return Tensor(resultBase);
}
} // namespace ty