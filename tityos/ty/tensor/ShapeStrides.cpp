#include "tityos/ty/tensor/ShapeStrides.h"

#include <iostream>

namespace ty {
namespace internal {
    ShapeStrides::ShapeStrides(const std::array<size_t, MAX_DIMS>& shape,
                               const std::array<size_t, MAX_DIMS>& strides,
                               size_t offset, size_t ndim)
        : shape_(shape), strides_(strides), offset_(offset), ndim_(ndim) {}

    ShapeStrides::ShapeStrides(const std::array<size_t, MAX_DIMS>& shape,
                               size_t ndim)
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

        for (size_t i = 0; i < ndim_; i++) {
            indexByte += (index % shape_[i]) * strides_[i];
            index /= shape_[i];
        }

        return indexByte * dtypeSize(dtype) + offset_;
    }

    std::array<size_t, MAX_DIMS>
    ShapeStrides::linearToTensorIndex(size_t linearIndex) const {
        std::array<size_t, MAX_DIMS> index{};

        for (size_t i = ndim_ - 1; i > 0; i--) {
            index[i] = linearIndex % shape_[i];
            linearIndex /= shape_[i];
        }
        index[0] = linearIndex % shape_[0];

        return index;
    }

    size_t ShapeStrides::tensorToLinearIndex(
        const std::array<size_t, MAX_DIMS>& index) const {
        size_t linear = 0;

        for (size_t i = 0; i < ndim_; i++) {
            linear += strides_[i] * index[i];
        }

        return linear;
    }

    size_t ShapeStrides::getNDim() const {
        return ndim_;
    }
    const std::array<size_t, MAX_DIMS>& ShapeStrides::getShape() const {
        return shape_;
    }
    const std::array<size_t, MAX_DIMS>& ShapeStrides::getStrides() const {
        return strides_;
    }
    size_t ShapeStrides::numElements() const {
        return std::accumulate(shape_.begin(), shape_.begin() + ndim_, 1,
                               std::multiplies<size_t>());
    }

    void ShapeStrides::initialStrides() {
        strides_ = std::array<size_t, MAX_DIMS>{};
        int currentStride = 1;

        for (int i = ndim_ - 1; i >= 0; i--) {
            strides_[i] = currentStride;
            currentStride *= shape_[i];
        }
    }
} // namespace internal
} // namespace ty