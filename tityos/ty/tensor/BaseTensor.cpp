#include "tityos/ty/tensor/BaseTensor.h"

#include <iostream>

namespace ty {
namespace internal {
    size_t BaseTensor::endIndex() const {
        return layout_.numElements(); 
    }

    BaseTensor::BaseTensor(std::shared_ptr<TensorStorage> data,
                           const ShapeStrides& layout, const DType dtype)
        : tensorStorage_(std::move(data)), layout_(layout), dtype_(dtype) {}

    BaseTensor::BaseTensor(const BaseTensor& other)
        : tensorStorage_(other.tensorStorage_), layout_(other.layout_),
          dtype_(other.dtype_) {}

    BaseTensor::BaseTensor(BaseTensor&& other) noexcept
        : tensorStorage_(std::move(other.tensorStorage_)),
          layout_(std::move(other.layout_)), dtype_(std::move(other.dtype_)) {}

    BaseTensor& BaseTensor::operator=(const BaseTensor& other) {
        if (this != &other) {
            tensorStorage_ = other.tensorStorage_;
            layout_ = other.layout_;
        }
        return *this;
    }

    BaseTensor& BaseTensor::operator=(BaseTensor&& other) noexcept {
        if (this != &other) {
            tensorStorage_ = std::move(other.tensorStorage_);
            layout_ = std::move(other.layout_);
        }
        return *this;
    }

    void* BaseTensor::at(const size_t* indexStart) const {
        size_t byteOffset = layout_.computeByteIndex(indexStart, dtype_);
        return tensorStorage_->at(byteOffset);
    }

    void* BaseTensor::at(size_t index) const {
        size_t byteOffset = layout_.computeByteIndex(index, dtype_);
        return tensorStorage_->at(byteOffset);
    }

    const ShapeStrides& BaseTensor::getLayout() const {
        return layout_;
    }

    const std::shared_ptr<TensorStorage>& BaseTensor::getTensorStorage() const {
        return tensorStorage_;
    }

    DType BaseTensor::getDType() const {
        return dtype_;
    }

    BaseTensor::Iterator::Iterator(
        const BaseTensor& baseTensor, size_t linearStartIndex)
        : baseTensor_(baseTensor), linearIndex_(linearStartIndex),
          ptr_(baseTensor.at(linearStartIndex)) {}

    void* BaseTensor::Iterator::operator->() {
        return ptr_;
    }

    void* BaseTensor::Iterator::operator*() {
        return ptr_;
    }

    // Prefix increment
    BaseTensor::Iterator& BaseTensor::Iterator::operator++() {
        linearIndex_++;
        ptr_ = baseTensor_.at(linearIndex_);
        return *this;
    }

    // Postfix increment
    BaseTensor::Iterator BaseTensor::Iterator::operator++(int) {
        Iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    size_t BaseTensor::Iterator::getIndex() const {
        return linearIndex_;
    }

    void BaseTensor::Iterator::jumpToIndex(
        size_t linearIndex) {
        linearIndex_ = linearIndex;
        ptr_ = baseTensor_.at(linearIndex_);
    }

    bool operator==(const BaseTensor::Iterator& a,
                    const BaseTensor::Iterator& b) {
        return (a.ptr_ == b.ptr_) && (a.linearIndex_ == b.linearIndex_);
    }

    bool operator!=(const BaseTensor::Iterator& a,
                    const BaseTensor::Iterator& b) {
        return (a.ptr_ != b.ptr_) || (a.linearIndex_ != b.linearIndex_);
    }

    BaseTensor::Iterator BaseTensor::begin() {
        return Iterator(*this, 0);
    }

    BaseTensor::Iterator BaseTensor::end() {
        return Iterator(*this, endIndex());
    }

    BaseTensor::Iterator BaseTensor::begin() const {
        return Iterator(*this, 0);
    }

    BaseTensor::Iterator BaseTensor::end() const {
        return Iterator(*this, endIndex());
    }
} // namespace internal
} // namespace ty