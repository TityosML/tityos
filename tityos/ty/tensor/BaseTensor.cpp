#include "tityos/ty/tensor/BaseTensor.h"

namespace ty {
namespace internal {
    const std::array<size_t, MAX_DIMS> BaseTensor::endIndex() const {
        // Exclusive end index
        // TODO: move into utils class to avoid duplicated code
        const size_t nDim = layout_.getNDim();
        const std::array<size_t, MAX_DIMS> shape = layout_.getShape();
        std::array<size_t, MAX_DIMS> endIdx{};

        for (size_t i = 0; i < nDim - 1; i++) {
            endIdx[i] = shape[i] - 1;
        }

        endIdx[nDim - 1] = shape[nDim - 1];
        return endIdx;
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
        size_t byteOffset = layout_.computeByteIndex(indexStart);
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

    const std::array<size_t, MAX_DIMS> BaseTensor::Iterator::endIndex() const {
        // Exclusive end index
        // TODO: move into utils class to avoid duplicated code
        const size_t nDim = baseTensor_.getLayout().getNDim();
        const std::array<size_t, MAX_DIMS> shape =
            baseTensor_.getLayout().getShape();
        std::array<size_t, MAX_DIMS> endIdx{};

        for (size_t i = 0; i < nDim - 1; i++) {
            endIdx[i] = shape[i] - 1;
        }

        endIdx[nDim - 1] = shape[nDim - 1];
        return endIdx;
    }

    void BaseTensor::Iterator::incrementIndex() {
        const std::array<size_t, internal::MAX_DIMS>& shape =
            baseTensor_.getLayout().getShape();
        const std::array<size_t, internal::MAX_DIMS>& strides =
            baseTensor_.getLayout().getStrides();
        const size_t nDim = baseTensor_.getLayout().getNDim();
        for (int i = nDim - 1; i >= 0; i--) {
            index_[i]++;
            if (index_[i] < shape[i]) {
                ptr_ = reinterpret_cast<char*>(ptr_) + strides[i];
                return;
            } else {
                index_[i] = 0;
                ptr_ =
                    reinterpret_cast<char*>(ptr_) - (shape[i] - 1) * strides[i];
            }
        }
        // Handle overflow as 1 more in last dimension
        index_ = endIndex();
        for (int i = nDim - 1; i >= 0; i--) {
            ptr_ = reinterpret_cast<char*>(ptr_) + strides[i] * (shape[i] - 1);
        }
        ptr_ = reinterpret_cast<char*>(ptr_) + strides[nDim - 1];
    }

    BaseTensor::Iterator::Iterator(
        const BaseTensor& baseTensor,
        const std::array<size_t, MAX_DIMS> startIndex)
        : baseTensor_(baseTensor), index_(startIndex),
          ptr_(baseTensor.at(startIndex.data())) {}

    void* BaseTensor::Iterator::operator->() {
        return ptr_;
    }

    void* BaseTensor::Iterator::operator*() {
        return ptr_;
    }

    // Prefix increment
    BaseTensor::Iterator& BaseTensor::Iterator::operator++() {
        incrementIndex();
        return *this;
    }

    // Postfix increment
    BaseTensor::Iterator BaseTensor::Iterator::operator++(int) {
        Iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const BaseTensor::Iterator& a,
                    const BaseTensor::Iterator& b) {
        return a.ptr_ == b.ptr_;
    }

    bool operator!=(const BaseTensor::Iterator& a,
                    const BaseTensor::Iterator& b) {
        return a.ptr_ != b.ptr_;
    }

    BaseTensor::Iterator BaseTensor::begin() {
        return Iterator(*this, std::array<size_t, MAX_DIMS>{});
    }

    BaseTensor::Iterator BaseTensor::end() {
        return Iterator(*this, endIndex());
    }

    BaseTensor::Iterator BaseTensor::begin() const {
        return Iterator(*this, std::array<size_t, MAX_DIMS>{});
    }

    BaseTensor::Iterator BaseTensor::end() const {
        return Iterator(*this, endIndex());
    }
} // namespace internal
} // namespace ty