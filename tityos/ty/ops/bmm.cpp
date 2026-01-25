#include "tityos/ty/ops/bmm.h"

#include <iostream>

namespace ty {
Tensor bmm(const Tensor& tensor1, const Tensor& tensor2) {

    auto b = internal::backend::getBackend(tensor1.getDevice().type());
    auto resultBase = std::make_shared<internal::BaseTensor>(
        b->bmm(*tensor1.getBaseTensor(),
               *tensor2.getBaseTensor()));

    return Tensor(resultBase);
}
} // namespace ty