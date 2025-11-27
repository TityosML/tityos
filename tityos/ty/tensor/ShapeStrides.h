#pragma once

#include "tityos/ty/tensor/Dtype.h"

#include <array>
#include <cstddef>

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

    ShapeStrides(const std::array<size_t, MAX_DIMS>& shape, DType dtype,
                 size_t ndim);

    size_t computeByteIndex(const size_t* indexStart) const;

    size_t getNDim() const;
    const std::array<size_t, MAX_DIMS>& getShape() const;
    const std::array<size_t, MAX_DIMS>& getStrides() const;

  private:
    void initialStrides(DType dtype);
};
} // namespace internal
} // namespace ty