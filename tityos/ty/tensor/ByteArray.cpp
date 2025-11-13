#include "tityos/ty/tensor/ByteArray.h"

namespace ty {
    namespace internal {
        void ByteArray::allocate() {
            if (device_.isCpu()) {
                startPointer_ = std::malloc(size_);
                if (!startPointer_) {
                    // TODO: Logging
                }
                return;
            }

            if (device_.isCuda()) {
                // TODO: CUDA allocation
                return;
            }

            // TODO: Unsupported device type
        }

        void ByteArray::deallocate() {
            if (device_.isCpu()) {
                std::free(startPointer_);
                startPointer_ = nullptr;
                return;
            }

            if (device_.isCuda()) {
                // TODO: CUDA deallocation
                return;
            }

            // TODO: Unsupported device type
        }
    } // namespace internal
} // namespace ty