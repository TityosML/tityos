#include "tityos/ty/backend/CPUBackend.h"

namespace ty {
namespace internal {
    namespace backend {
        static CPUBackend cpuBackendInstance;
    } // namespace backend
} // namespace internal
} // namespace ty

extern "C" {
TITYOS_API ty::internal::backend::Backend* registerCpuBackend() {
    return &ty::internal::backend::cpuBackendInstance;
}
}