#include "tityos/ty/ops/cpu/toCpu.h"

namespace ty {
namespace internal {
    BaseTensor backend::CPUBackend::toCpu(const BaseTensor& tensor) {
        // Potentially make copy? to be decided
        return tensor;
    }
}; // namespace internal
} // namespace ty