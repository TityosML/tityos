#include "tityos/ty/tensor/TensorStorage.h"

namespace ty {
    namespace internal {
        void TensorStorage::allocate() {
            if (device_.isCpu()) {
                startPointer_ = std::malloc(size_);
                if (!startPointer_) {
                    // TODO: Logging
                }
                return;
            }

            if (device_.isCuda()) {
                if (cuda::isCudaAvailable()) {
                    #ifdef TITYOS_USE_CUDA
                        // TODO: CUDA allocation
                    #endif
                } else {
                    #ifdef TITYOS_USE_CUDA
                        // TODO: Error, CUDA device not found
                    #else
                        // TODO: Error, not built with CUDA
                    #endif
                }
                return;
            }

            // TODO: Unsupported device type
        }

        void TensorStorage::deallocate() {
            if (device_.isCpu()) {
                std::free(startPointer_);
                startPointer_ = nullptr;
                return;
            }

            if (device_.isCuda()) {
                #ifdef TITYOS_USE_CUDA
                    // TODO: CUDA deallocation
                #endif
                return;
            }

            // TODO: Unsupported device type
        }
    } // namespace internal
} // namespace ty