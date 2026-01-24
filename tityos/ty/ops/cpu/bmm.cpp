#include "tityos/ty/ops/cpu/bmm.h"

namespace ty {
namespace internal {
    BaseTensor backend::CPUBackend::bmm(const BaseTensor& batch1,
                                        const BaseTensor& batch2) {
        return batch1;
    }
}; // namespace internal
} // namespace ty