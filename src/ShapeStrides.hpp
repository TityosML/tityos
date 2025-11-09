#pragma once
#include <array>
#include <cstddef>

constexpr size_t MAX_DIMS = 64;

class ShapeStrides {
    private:
        std::array<size_t, MAX_DIMS> shape;
        std::array<size_t, MAX_DIMS> strides;
        size_t offset;
        size_t ndim;

    public:
        ShapeStrides(
            const std::array<size_t, MAX_DIMS>& shape_,
            const std::array<size_t, MAX_DIMS>& strides_,
            size_t offset_,
            size_t ndim_
        ) : shape(shape_), strides(strides_), offset(offset_), ndim(ndim_) {}

        size_t computeByteIndex(const std::array<size_t, MAX_DIMS>& index) const {
            size_t byteIndex = offset;
            for (size_t i = 0; i < ndim; ++i) {
                byteIndex += index[i] * strides[i];
            }
            return byteIndex;
        }

        size_t getNDim() const { return ndim; }
        const std::array<size_t, MAX_DIMS>& getShape() const { return shape; }
        const std::array<size_t, MAX_DIMS>& getStrides() const { return strides; }
};