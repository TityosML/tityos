#pragma once

#include <ranges>
#include <span>
#include <string>

#include "tityos/ty/export.h"
#include "tityos/ty/tensor/BaseTensor.h"

namespace ty {
    TITYOS_EXPORT class Tensor {
      private:
        std::shared_ptr<internal::BaseTensor> baseTensor_;

      public:
        template <std::ranges::contiguous_range R>
        Tensor(const R &data, std::span<const size_t> shape, Device device = {DeviceType::CPU, 0},
               DType dtype = DType::Float32) {
            using T = std::ranges::range_value_t<R>;

            size_t numElements = std::ranges::size(data);
            size_t numBytes = sizeof(T) * numElements;
            std::shared_ptr<internal::TensorStorage> dataStorage =
                std::make_shared<internal::TensorStorage>(numBytes, device);
            std::memcpy(dataStorage->begin(), std::ranges::data(data), numBytes);

            std::array<size_t, internal::MAX_DIMS> storageShape{};
            int i = 0;
            for (size_t s : shape) {
                storageShape[i++] = s;
            }

            internal::ShapeStrides layout(storageShape, dtype, shape.size());

            baseTensor_ = std::make_shared<internal::BaseTensor>(dataStorage, layout);
        }

        template <std::ranges::contiguous_range R>
        Tensor(const R &data, const std::initializer_list<size_t> &shape,
               Device device = {DeviceType::CPU, 0}, DType dtype = DType::Float32)
            : Tensor(data, std::span<const size_t>(shape.begin(), shape.size()), device, dtype) {}

        template <std::ranges::contiguous_range R>
        Tensor(const R &data, const std::vector<size_t> &shape,
               Device device = {DeviceType::CPU, 0}, DType dtype = DType::Float32)
            : Tensor(data, std::span<const size_t>(shape.data(), shape.size()), device, dtype) {}

        template <std::ranges::contiguous_range R, size_t N>
        Tensor(const R &data, const std::array<size_t, N> &shape,
               Device device = {DeviceType::CPU, 0}, DType dtype = DType::Float32)
            : Tensor(data, std::span<const size_t>(shape.data(), shape.size()), device, dtype) {}

        template <size_t N>
        void *at(const std::array<size_t, N> &index) const {
            // TODO : Check dimension and index
            return baseTensor_->at(index.data());
        }

        void *at(const std::vector<size_t> &index) const {
            // TODO : Check dimension and index
            return baseTensor_->at(index.data());
        }

        void *at(const std::initializer_list<size_t> &index) const {
            // TODO : Check dimension and index
            return baseTensor_->at(index.begin());
        }

        std::string toString() const {
            return "";
        }
    };
} // namespace ty