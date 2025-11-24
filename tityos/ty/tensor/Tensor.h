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
        
        inline void *at(const size_t *indexStart, const size_t indexSize) const {
            if (indexSize != baseTensor_->getLayout().getNDim()) {
                throw std::invalid_argument("Index size mismatch: expected "
                + std::to_string(baseTensor_->getLayout().getNDim())
                + ", got " + std::to_string(indexSize));
            }

            const std::array<size_t, internal::MAX_DIMS>& shape = baseTensor_->getLayout().getShape();
            for (size_t i = 0; i < indexSize; i++) {
                if (indexStart[i] >= shape[i]) {
                    throw std::out_of_range("Index out of bounds at dimension " + std::to_string(i) 
                    + ": index value " + std::to_string(*(indexStart + i))
                    + " is >= shape dimension " + std::to_string(shape[i]));
                }
            }

            return baseTensor_->at(indexStart);
        }

        template <size_t N>
        inline void *at(const std::array<size_t, N> &index) const {
            return at(index.data(), N);
        }

        inline void *at(const std::vector<size_t> &index) const {
            return at(index.data(), index.size());
        }

        inline void *at(const std::initializer_list<size_t> &index) const {
            return at(index.begin(), index.size());
        }
        
        using Iterator = internal::BaseTensor::Iterator;
        
        Iterator begin() { return baseTensor_->begin(); }
        Iterator end() { return baseTensor_->end(); }

        Iterator begin() const { return baseTensor_->begin(); }
        Iterator end() const { return baseTensor_->end(); }

        std::string toString() const {
            // TODO: Add [] formatting and deal with datatypes
            std::string str = "";
            for (auto it = begin(); it != end(); it++){
                str += std::to_string(*static_cast<float*>(*it));
            }
            str += std::to_string(*static_cast<float*>(*end()));
            return str;
        }
    };
} // namespace ty