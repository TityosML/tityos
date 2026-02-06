#include "tityos/ty/tensor/cpu/TensorStorage.h"

namespace ty {
namespace internal {
    void* backend::CPUBackend::allocate(size_t bytes,
                                        [[maybe_unused]] int index) {
        void* ptr = std::malloc(bytes);

        if (!ptr) {
            throw std::runtime_error("Unable to allocate memory for CPU");
        }

        return ptr;
    }

    void backend::CPUBackend::deallocate(void* ptr) {
        std::free(ptr);
    }

    void backend::CPUBackend::copyData(void* destPtr, const void* srcPtr,
                                       size_t numBytes) {
        std::memcpy(destPtr, srcPtr, numBytes);
    }

    void backend::CPUBackend::copyDataFromCpu(void* destPtr, const void* srcPtr,
                                              size_t numBytes) {
        std::memcpy(destPtr, srcPtr, numBytes);
    }
} // namespace internal
} // namespace ty