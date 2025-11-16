#pragma once

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
            ShapeStrides(const std::array<size_t, MAX_DIMS> &shape,
                         const std::array<size_t, MAX_DIMS> &strides, size_t offset, size_t ndim)
                : shape_(shape), strides_(strides), offset_(offset), ndim_(ndim) {}

            size_t computeByteIndex(const std::array<size_t, MAX_DIMS> &index) const {
                size_t byteIndex = offset_;
                for (size_t i = 0; i < ndim_; i++) {
                    byteIndex += index[i] * strides_[i];
                }
                return byteIndex;
            }

            size_t getNDim() const {
                return ndim_;
            }
            const std::array<size_t, MAX_DIMS> &getShape() const {
                return shape_;
            }
            const std::array<size_t, MAX_DIMS> &getStrides() const {
                return strides_;
            }
        };
    } // namespace internal
} // namespace ty