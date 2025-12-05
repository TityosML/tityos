#pragma once

#include "tityos/ty/tensor/Dtype.h"

#include <array>
#include <cstddef>
#include <numeric>

namespace ty {
namespace internal {
    constexpr size_t MAX_DIMS = 64;

    class ShapeStrides {
      private:
        std::array<size_t, MAX_DIMS> shape_;
        std::array<size_t, MAX_DIMS> strides_;

        size_t offset_;
        size_t ndim_;

      public:
        ShapeStrides(const std::array<size_t, MAX_DIMS>& shape,
                     const std::array<size_t, MAX_DIMS>& strides, size_t offset,
                     size_t ndim);

        ShapeStrides(const std::array<size_t, MAX_DIMS>& shape, size_t ndim);

        size_t computeByteIndex(const size_t* indexStart, DType dtype) const;
        size_t computeByteIndex(size_t index, DType dtype) const;

        std::array<size_t, MAX_DIMS>
        linearToTensorIndex(size_t linearIndex) const;
        size_t tensorToLinearIndex(
            const std::array<size_t, MAX_DIMS>& linearIndex) const;

        size_t getNDim() const;
        const std::array<size_t, MAX_DIMS>& getShape() const;
        const std::array<size_t, MAX_DIMS>& getStrides() const;
        size_t numElements() const;

        bool operator==(const ShapeStrides& other) const;

      private:
        void initialStrides();
    };
} // namespace internal
} // namespace ty