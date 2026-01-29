#pragma once

#include "tityos/ty/tensor/Dtype.h"

#include <array>
#include <cstddef>
#include <numeric>
#include <optional>
#include <type_traits>

namespace ty {
namespace internal {
    constexpr size_t MAX_DIMS = 64;
    using TensorStrides = std::array<ptrdiff_t, MAX_DIMS>;
    using TensorShape = std::array<size_t, MAX_DIMS>;

    class ShapeStrides {
      private:
        TensorShape shape_;
        TensorStrides strides_;

        size_t offset_;
        size_t ndim_;

      public:
        ShapeStrides(const TensorShape& shape, const TensorStrides& strides,
                     size_t offset, size_t ndim);

        ShapeStrides(const TensorShape& shape, size_t ndim);

        size_t computeByteIndex(const size_t* indexStart, DType dtype) const;
        size_t computeByteIndex(size_t index, DType dtype) const;

        std::array<size_t, MAX_DIMS>
        linearToTensorIndex(size_t linearIndex) const;
        size_t tensorToLinearIndex(
            const std::array<size_t, MAX_DIMS>& linearIndex) const;

        const TensorShape& getShape() const;
        const TensorStrides& getStrides() const;
        size_t getNDim() const;
        size_t getOffset() const;
        size_t numElements() const;

        bool isContiguous() const;

        bool operator==(const ShapeStrides& other) const;

        ShapeStrides slice(size_t dim,
                           std::optional<ptrdiff_t> start = std::nullopt,
                           std::optional<ptrdiff_t> stop = std::nullopt,
                           ptrdiff_t step = 1) const;

        ShapeStrides select(size_t dim, ptrdiff_t select) const;

      private:
        void initialStrides();
    };
} // namespace internal
} // namespace ty