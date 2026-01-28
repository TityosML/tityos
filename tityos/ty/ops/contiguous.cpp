#include "tityos/ty/ops/contiguous.h"

namespace ty {
Tensor contiguous(const Tensor& tensor) {
    auto b = internal::backend::getBackend(tensor.getDevice().type());
    auto resultBase = std::make_shared<internal::BaseTensor>(
        b->contiguous(*tensor.getBaseTensor()));
    return Tensor(resultBase);
}
} // namespace ty