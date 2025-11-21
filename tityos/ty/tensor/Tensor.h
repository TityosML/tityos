#pragma once

#include <ranges>
#include <string>

#include "tityos/ty/export.h"
#include "tityos/ty/tensor/BaseTensor.h"

namespace ty {
    TITYOS_EXPORT class Tensor {
      private:
        std::shared_ptr<internal::BaseTensor> baseTensor_;

      public:
        template <std::ranges::contiguous_range R>
        Tensor(const R &data, const std::initializer_list<size_t> &shape,
               Device device = {DeviceType::CPU, 0}, DType dtype = DType::Float32) {
            using T = std::ranges::range_value_t<R>;

            size_t numElements = std::ranges::size(data);
            size_t numBytes = sizeof(T) * numElements;
            std::shared_ptr<internal::TensorStorage> dataStorage = std::make_shared<internal::TensorStorage>(numBytes, device);
            std::memcpy(dataStorage->begin(), std::ranges::data(data), numBytes);

            std::array<size_t, internal::MAX_DIMS> storageShape{};
            int i = 0;
            for (size_t s : shape) {
                storageShape[i++] = s;
            }

            internal::ShapeStrides layout(storageShape, dtype, shape.size());

            baseTensor_ = std::make_shared<internal::BaseTensor>(dataStorage, layout);
        }

        void *at(const std::array<size_t, internal::MAX_DIMS> &index) const {
            // TODO: Not efficient to create new array everytime
            //       Need to add option to subclasses to allow for std::initializer_list and
            //       std::vector

            std::array<size_t, internal::MAX_DIMS> arrayIndex{};
            for (int i = 0; i < index.size(); i++) {
                arrayIndex[i] = index[i];
            }

            return baseTensor_->at(arrayIndex);
        }

        std::string toString() const {
            return "";
        }
    };
} // namespace ty