#include "tityos/ty/ops/contiguous.h"

namespace ty {
namespace internal {
    BaseTensor contiguous(const BaseTensor& tensor) {
        auto b = internal::backend::getBackend(tensor.getDevice().type());
        return b->contiguous(tensor);
    }
} // namespace internal

Tensor contiguous(const Tensor& tensor) {
    return Tensor(std::make_shared<internal::BaseTensor>(internal::contiguous(*tensor.getBaseTensor())));
}
} // namespace ty