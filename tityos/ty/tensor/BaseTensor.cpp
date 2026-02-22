#include "tityos/ty/tensor/BaseTensor.h"

namespace ty {
namespace internal {
    size_t BaseTensor::endIndex() const {
        return layout_.numElements();
    }

    BaseTensor::BaseTensor(std::shared_ptr<TensorStorage> data,
                           const ShapeStrides& layout, const DType dtype)
        : tensorStorage_(std::move(data)), layout_(layout), dtype_(dtype) {}

    BaseTensor BaseTensor::copy() const {
        return BaseTensor(std::make_shared<TensorStorage>(*tensorStorage_),
                          layout_, dtype_);
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

    size_t BaseTensor::getNDim() const {
        return layout_.getNDim();
    }

    const TensorShape& BaseTensor::getShape() const {
        return layout_.getShape();
    }

    const TensorStrides& BaseTensor::getStrides() const {
        return layout_.getStrides();
    }

    size_t BaseTensor::getLogicalSize() const {
        return layout_.numElements() * dtypeSize(dtype_);
    }

    size_t BaseTensor::getSize() const {
        return tensorStorage_->getSize();
    }

    const std::shared_ptr<TensorStorage>& BaseTensor::getTensorStorage() const {
        return tensorStorage_;
    }

    Device BaseTensor::getDevice() const {
        return tensorStorage_->getDevice();
    }

    DType BaseTensor::getDType() const {
        return dtype_;
    }

    bool BaseTensor::isContiguous() const {
        return layout_.isContiguous();
    }

    bool BaseTensor::operator==(const BaseTensor& other) const {
        return tensorStorage_ == other.tensorStorage_ &&
               layout_ == other.layout_ && dtype_ == other.dtype_;
    }

    BaseTensor BaseTensor::slice(size_t dim, std::optional<ptrdiff_t> start,
                                 std::optional<ptrdiff_t> stop,
                                 ptrdiff_t step) const {
        return BaseTensor(tensorStorage_, layout_.slice(dim, start, stop, step),
                          dtype_);
    };

    BaseTensor BaseTensor::indexList(IndexList indices) const {
        internal::IndexingResult idxResult =
            internal::resolveIndices(indices, *this);

        if (!idxResult.gather.has_value()) {
            return BaseTensor(tensorStorage_, idxResult.newLayout, dtype_);
        } else {
            return copyFromGather(*this, idxResult);
        }
    }

    BaseTensor::Iterator::Iterator(const BaseTensor& baseTensor,
                                   size_t linearStartIndex)
        : baseTensor_(baseTensor), linearIndex_(linearStartIndex),
          ptr_(baseTensor.at(linearStartIndex)) {}

    void* BaseTensor::Iterator::operator->() {
        return ptr_;
    }

    void* BaseTensor::Iterator::operator*() {
        return ptr_;
    }

    BaseTensor::Iterator& BaseTensor::Iterator::operator++() {
        linearIndex_++;
        ptr_ = baseTensor_.at(linearIndex_);
        return *this;
    }

    BaseTensor::Iterator BaseTensor::Iterator::operator++(int) {
        Iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    size_t BaseTensor::Iterator::getIndex() const {
        return linearIndex_;
    }

    void BaseTensor::Iterator::jumpToIndex(size_t linearIndex) {
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