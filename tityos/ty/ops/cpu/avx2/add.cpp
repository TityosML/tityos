#include "tityos/ty/ops/cpu/avx2/add.h"

#include "tityos/ty/ops/cpu/add.h"

namespace ty {
namespace internal {
    BaseTensor internalAddAvx2(const BaseTensor& tensor1,
                               const BaseTensor& tensor2) {
        // TODO: Implment avx2 addition
        return tensor1;
    }
} // namespace internal
} // namespace ty
