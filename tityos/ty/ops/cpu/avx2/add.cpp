#include "tityos/ty/ops/cpu/avx2/add.h"

#include "tityos/ty/ops/cpu/add.h"

namespace ty {
namespace internal {
    void internalAddAvx2(Tensor& result, const Tensor& tensor1,
                         const Tensor& tensor2) {
        // TODO: Implment avx2 addition
        internalAddCpu(result, tensor1, tensor2);
    }
} // namespace internal
} // namespace ty
