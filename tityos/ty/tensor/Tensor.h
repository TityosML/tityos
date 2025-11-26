#pragma once

#include "tityos/ty/export.h"
#include "tityos/ty/tensor/BaseTensor.h"
#include <cstring>
#include <iterator>
#include <string>

namespace ty {
    TITYOS_EXPORT class Tensor {
      private:
        std::shared_ptr<internal::BaseTensor> baseTensor_;

      public:
        template <
            class DataContainer, class ShapeContainer,
            std::enable_if_t<std::is_same_v<typename ShapeContainer::value_type, size_t> &&
                                 std::is_trivially_copyable_v<typename DataContainer::value_type>,
                             int> = 0>
        Tensor(const DataContainer &data, const ShapeContainer &shape,
               Device device = {DeviceType::CPU, 0}, DType dtype = DType::Float32) {
            using T = typename DataContainer::value_type;

            if (std::size(shape) > internal::MAX_DIMS) {
                throw std::runtime_error("Tensor shape has too many dimensions.");
            }

            size_t numElements = std::size(data);
            size_t numBytes = sizeof(T) * numElements;
            std::shared_ptr<internal::TensorStorage> dataStorage =
                std::make_shared<internal::TensorStorage>(numBytes, device);
            std::memcpy(dataStorage->begin(), std::data(data), numBytes);

            std::array<size_t, internal::MAX_DIMS> storageShape{};
            int i = 0;
            for (size_t s : shape) {
                storageShape[i++] = s;
            }

            internal::ShapeStrides layout(storageShape, dtype, std::size(shape));

            baseTensor_ = std::make_shared<internal::BaseTensor>(dataStorage, layout);
        }

        template <typename T, class ShapeContainer>
        Tensor(std::initializer_list<T> data, const ShapeContainer &shape)
            : Tensor(std::vector<T>(data), shape) {}

        template <class DataContainer>
        Tensor(const DataContainer &data, std::initializer_list<size_t> shape)
            : Tensor(data, std::vector<size_t>(shape)) {}

        template <typename T>
        Tensor(std::initializer_list<T> data, std::initializer_list<size_t> shape)
            : Tensor(std::vector<T>(data), std::vector<size_t>(shape)) {}

        Tensor(const Tensor &other) : baseTensor_(other.baseTensor_) {}

        Tensor(Tensor &&other) noexcept : baseTensor_(std::move(other.baseTensor_)) {}

        Tensor &operator=(const Tensor &other) {
            if (this != &other) {
                baseTensor_ = other.baseTensor_;
            }
            return *this;
        }

        Tensor &operator=(Tensor &&other) noexcept {
            if (this != &other) {
                baseTensor_ = std::move(other.baseTensor_);
            }
            return *this;
        }

        void *at(const size_t *indexStart, const size_t indexSize) const;

        template <size_t N> void *at(const std::array<size_t, N> &index) const {
            return at(index.data(), N);
        }

        void *at(const std::vector<size_t> &index) const {
            return at(index.data(), index.size());
        }

        void *at(const std::initializer_list<size_t> &index) const {
            return at(index.begin(), index.size());
        }

        using Iterator = internal::BaseTensor::Iterator;

        Iterator begin() {
            return baseTensor_->begin();
        }
        Iterator end() {
            return baseTensor_->end();
        }

        Iterator begin() const {
            return baseTensor_->begin();
        }

        Iterator end() const {
            return baseTensor_->end();
        }

        std::string toString() const;
    };
} // namespace ty