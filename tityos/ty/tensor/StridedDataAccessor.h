#pragma once

#include "tityos/ty/tensor/ByteArray.h"
#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <memory>

namespace ty {
    namespace internal {
        class StridedDataAccessor {
          private:
            std::shared_ptr<ByteArray> byteArray_;
            ShapeStrides layout_;

          public:
            StridedDataAccessor(std::shared_ptr<ByteArray> data, const ShapeStrides &layout)
                : byteArray_(std::move(data)), layout_(layout) {}

            void *at(const std::array<size_t, MAX_DIMS> &index) const {
                size_t byteOffset = layout_.computeByteIndex(index);
                return byteArray_->at(byteOffset);
            }

            const ShapeStrides &getLayout() const {
                return layout_;
            }
            const std::shared_ptr<ByteArray> &getByteArray() const {
                return byteArray_;
            }
        };
    } // namespace internal
} // namespace ty