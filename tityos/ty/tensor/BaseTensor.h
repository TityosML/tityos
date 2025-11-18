#pragma once

#include <memory>

#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"
#include "tityos/ty/tensor/TensorStorage.h"

namespace ty {
    namespace internal {
        class BaseTensor {
          private:
            DType dtype_;

            std::shared_ptr<TensorStorage> byteArray_;
            ShapeStrides layout_;
          
          public:
            BaseTensor(std::shared_ptr<TensorStorage> data, const ShapeStrides &layout)
                : byteArray_(std::move(data)), layout_(layout) {}

            void *at(const std::array<size_t, MAX_DIMS> &index) const {
                size_t byteOffset = layout_.computeByteIndex(index);
                return byteArray_->at(byteOffset);
            }

            const ShapeStrides &getLayout() const {
                return layout_;
            }
            const std::shared_ptr<TensorStorage> &getByteArray() const {
                return byteArray_;
            }
        };
    } // namespace internal
} // namespace ty