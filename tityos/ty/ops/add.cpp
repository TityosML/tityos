#include "tityos/ty/ops/add.h"

namespace ty {
Tensor add(const Tensor& tensor1, const Tensor& tensor2) {
    auto broadcastShape =
        internal::broadcastShape(tensor1.getShape(), tensor1.getNDim(),
                                 tensor2.getShape(), tensor2.getNDim());

    auto broadcastedTensor1 = expand(tensor1, broadcastShape);
    auto broadcastedTensor2 = expand(tensor2, broadcastShape);

    if (tensor1.getDType() != tensor2.getDType()) {
        throw std::invalid_argument("Types must match for addition");
    }

    auto b = internal::backend::getBackend(tensor1.getDevice().type());
    auto resultBase = std::make_shared<internal::BaseTensor>(
        b->add(*broadcastedTensor1.getBaseTensor(),
               *broadcastedTensor2.getBaseTensor()));

    auto result = Tensor(resultBase);

    auto contextStorage = internal::GradientContextStorage(tensor1, tensor2);
    auto context = internal::GradientContext(
        contextStorage, [](const internal::GradientContextStorage& tensors) {
            tensors[0].addGrad(*tensors[1].getBaseTensor());
            tensors[1].addGrad(*tensors[0].getBaseTensor());
        });

    result.setGradContext(context);

    return result;
}
} // namespace ty