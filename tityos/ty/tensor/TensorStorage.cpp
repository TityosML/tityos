#include "tityos/ty/tensor/TensorStorage.h"

namespace ty {
namespace internal {
    TensorStorage::TensorStorage(size_t numBytes, Device device)
        : size_(numBytes), device_(device) {
        allocate();
    }

    TensorStorage::~TensorStorage() {
        deallocate();
    }

    void* TensorStorage::at(size_t index) {
        return reinterpret_cast<char*>(startPointer_) + index;
    }

    size_t TensorStorage::getSize() const {
        return size_;
    }

    Device TensorStorage::getDevice() const {
        return device_;
    }

    void* TensorStorage::begin() {
        return startPointer_;
    }

    void* TensorStorage::end() {
        return reinterpret_cast<char*>(startPointer_) + size_;
    }

    void TensorStorage::allocate() {
        if (device_.isCpu()) {
            startPointer_ = std::malloc(size_);
            if (!startPointer_) {
                // TODO: Logging
            }
            return;
        }

        if (device_.isCuda()) {
#ifdef TITYOS_USE_CUDA
            if (cuda::isCudaAvailable()) {
                cudaMalloc(&startPointer_, size_);
            } else {
                throw std::runtime_error("Cannot allocate Tensor Data to CUDA. "
                                         "CUDA is not available");
            }
#else
            throw std::runtime_error("Cannot allocate Tensor Data to CUDA. "
                                     "Tityos not built with CUDA");
#endif
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
            cudaFree(startPointer_);
            startPointer_ = nullptr;
#endif
            return;
        }

        // TODO: Unsupported device type
    }
} // namespace internal
} // namespace ty