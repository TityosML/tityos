#include "tityos/ty/tensor/ShapeStrides.h"

namespace ty {
namespace internal {
    ShapeStrides::ShapeStrides(const std::array<size_t, MAX_DIMS>& shape,
                               const std::array<size_t, MAX_DIMS>& strides,
                               size_t offset, size_t ndim)
        : shape_(shape), strides_(strides), offset_(offset), ndim_(ndim) {}

    ShapeStrides::ShapeStrides(const std::array<size_t, MAX_DIMS>& shape,
                               DType dtype, size_t ndim)
        : shape_(shape), offset_(0), ndim_(ndim) {
        initialStrides(dtype);
    }

    size_t ShapeStrides::computeByteIndex(const size_t* indexStart) const {
        size_t byteIndex = offset_;
        for (size_t i = 0; i < ndim_; i++) {
            byteIndex += *indexStart * strides_[i];
            indexStart++;
        }
        return byteIndex;
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

    void ShapeStrides::initialStrides(DType dtype) {
        strides_ = std::array<size_t, MAX_DIMS>{};
        int currentStride = 1;

        for (int i = ndim_ - 1; i >= 0; i--) {
            strides_[i] = currentStride * dtypeSize(dtype);
            currentStride *= shape_[i];
        }
    }
} // namespace internal
} // namespace ty