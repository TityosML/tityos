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

    TensorStorage::TensorStorage(const TensorStorage& other)
        : size_(other.size_), device_(other.device_) {
        allocate();

        auto b = backend::getBackend(device_.type());
        b->copyData(startPointer_, other.startPointer_, size_);
    }

    TensorStorage::TensorStorage(TensorStorage&& other) noexcept
        : startPointer_(other.startPointer_), size_(other.size_),
          device_(other.device_) {
        other.size_ = 0;
        other.startPointer_ = nullptr;
    }

    TensorStorage& TensorStorage::operator=(const TensorStorage& other) {
        if (this == &other) {
            return *this;
        }

        deallocate();

        size_ = other.size_;
        device_ = other.device_;

        allocate();

        auto b = backend::getBackend(device_.type());
        b->copyData(startPointer_, other.startPointer_, size_);

        return *this;
    }

    TensorStorage& TensorStorage::operator=(TensorStorage&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        deallocate();

        size_ = other.size_;
        device_ = other.device_;
        startPointer_ = other.startPointer_;

        other.size_ = 0;
        other.startPointer_ = nullptr;

        return *this;
    }

    void TensorStorage::copyDataFromCpu(const void* dataStartPointer,
                                        size_t numBytes) {
        auto b = backend::getBackend(device_.type());
        b->copyDataFromCpu(startPointer_, dataStartPointer, numBytes);
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
        auto b = backend::getBackend(device_.type());
        startPointer_ = b->allocate(size_, device_.index());

        if (!startPointer_) {
            throw std::runtime_error(
                "Unable to allocate memory to TensorStorage");
        }
    }

    void TensorStorage::deallocate() {
        auto b = backend::getBackend(device_.type());
        b->deallocate(startPointer_);
        startPointer_ = nullptr;
    }
} // namespace internal
} // namespace ty