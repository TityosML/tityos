#include "tityos/ty/tensor/ShapeStrides.h"

#include "tityos/ty/tensor/Tensor.h"

#include <algorithm>

namespace ty {
namespace internal {
    ShapeStrides::ShapeStrides(const TensorShape& shape,
                               const TensorStrides& strides, size_t offset,
                               size_t ndim)
        : shape_(shape), strides_(strides), offset_(offset), ndim_(ndim) {}

    ShapeStrides::ShapeStrides(const TensorShape& shape, size_t ndim)
        : shape_(shape), offset_(0), ndim_(ndim) {
        initialStrides();
    }

    size_t ShapeStrides::computeByteIndex(const size_t* indexStart,
                                          DType dtype) const {
        size_t byteIndex = offset_;
        for (size_t i = 0; i < ndim_; i++) {
            byteIndex += *indexStart * strides_[i] * dtypeSize(dtype);
            indexStart++;
        }
        return byteIndex;
    }

    size_t ShapeStrides::computeByteIndex(size_t index, DType dtype) const {
        size_t indexByte = 0;

        for (size_t i = ndim_; i-- > 0;) {
            indexByte += (index % shape_[i]) * strides_[i];
            index /= shape_[i];
        }

        return indexByte * dtypeSize(dtype) + offset_;
    }

    std::array<size_t, MAX_DIMS>
    ShapeStrides::linearToTensorIndex(size_t linearIndex) const {
        std::array<size_t, MAX_DIMS> index{};

        for (size_t i = ndim_; i-- > 0;) {
            index[i] = linearIndex % shape_[i];
            linearIndex /= shape_[i];
        }

        return index;
    }

    size_t ShapeStrides::tensorToLinearIndex(
        const std::array<size_t, MAX_DIMS>& index) const {
        size_t linear = 0;
        size_t dimProduct = 1;

        for (size_t i = ndim_; i-- > 0;) {
            linear += dimProduct * index[i];
            dimProduct *= shape_[i];
        }

        return linear;
    }

    ShapeStrides ShapeStrides::slice(size_t dim, ptrdiff_t start,
                                     ptrdiff_t stop, ptrdiff_t step) const {
        if (dim >= ndim_) {
            throw std::out_of_range(
                "Slice dimension exceeds tensor dimensions");
        }

        if (step == 0) {
            throw std::invalid_argument("Step cannot be 0.");
        }

        ptrdiff_t size = static_cast<ptrdiff_t>(shape_[dim]);

        if (start < 0) {
            start += size;
        }

        if (stop < 0) {
            stop += size;
        }

        if (step > 0) {
            start = std::clamp(start, ptrdiff_t{0}, size);
            stop = std::clamp(stop, ptrdiff_t{0}, size);
        } else {
            start = std::clamp(start, ptrdiff_t{0}, size - 1);
            stop = std::clamp(stop, ptrdiff_t{-1}, size - 1);
        }

        ptrdiff_t newSize = std::max<ptrdiff_t>(
            0, (stop - start + step + (step > 0 ? -1 : 1)) / step);

        TensorShape newShape = shape_;
        newShape[dim] = static_cast<size_t>(newSize);

        TensorStrides newStrides = strides_;
        newStrides[dim] *= step;

        size_t newOffset = offset_ + static_cast<size_t>(start) *
                                         static_cast<size_t>(strides_[dim]);

        return ShapeStrides(newShape, newStrides, newOffset, ndim_);
    }

    const TensorShape& ShapeStrides::getShape() const {
        return shape_;
    }
    const TensorStrides& ShapeStrides::getStrides() const {
        return strides_;
    }
    size_t ShapeStrides::getNDim() const {
        return ndim_;
    }
    size_t ShapeStrides::getOffset() const {
        return offset_;
    }
    size_t ShapeStrides::numElements() const {
        return std::accumulate(shape_.begin(), shape_.begin() + ndim_, 1,
                               std::multiplies<size_t>());
    }

    void ShapeStrides::initialStrides() {
        strides_ = TensorStrides{};
        int currentStride = 1;

        for (int i = ndim_ - 1; i >= 0; i--) {
            strides_[i] = currentStride;
            currentStride *= shape_[i];
        }
    }

    bool ShapeStrides::operator==(const ShapeStrides& other) const {
        return ndim_ == other.ndim_ && shape_ == other.shape_ &&
               strides_ == other.strides_ && offset_ == other.offset_;
    }
} // namespace internal
} // namespace ty