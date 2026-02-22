#include "tityos/ty/ops/toCpu.h"

namespace ty {
Tensor toCpu(const Tensor& tensor) {
    auto b = internal::backend::getBackend(tensor.getDevice().type());
    auto resultBase = std::make_shared<internal::BaseTensor>(b->toCpu(*tensor.getBaseTensor()));
    return Tensor(resultBase);
}
} // namespace ty