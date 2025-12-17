#include "tityos/ty/ops/add.h"

namespace ty {
Tensor add(const Tensor& a, const Tensor& b) {
    Tensor result = a;

    internal::internalAdd(a.getDevice(), result, a, b);

    return result;
}
} // namespace ty