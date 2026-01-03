#include "tityos/ty/ops/add.h"

namespace ty {
Tensor add(const Tensor& tensor1, const Tensor& tensor2) {
    auto resultBase = std::make_shared<internal::BaseTensor>(
        internal::internalAdd(tensor1.getDevice(), *tensor1.getBaseTensor(),
                              *tensor2.getBaseTensor()));

    return Tensor(resultBase);
}
} // namespace ty