#include "tityos/ty/backend/CUDABackend.h"

namespace ty {
namespace internal {
    namespace backend {
        static CUDABackend cudaBackendInstance;
    } // namespace backend
} // namespace internal
} // namespace ty

extern "C" {
TITYOS_API ty::internal::backend::Backend* registerCudaBackend() {
    return &ty::internal::backend::cudaBackendInstance;
}
}