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

        inline void *at(const size_t *indexStart, const size_t indexSize) const {
            if (indexSize != baseTensor_->getLayout().getNDim()) {
                throw std::invalid_argument("Index size mismatch: expected " +
                                            std::to_string(baseTensor_->getLayout().getNDim()) +
                                            ", got " + std::to_string(indexSize));
            }

            const std::array<size_t, internal::MAX_DIMS> &shape =
                baseTensor_->getLayout().getShape();
            for (size_t i = 0; i < indexSize; i++) {
                if (indexStart[i] >= shape[i]) {
                    throw std::out_of_range("Index out of bounds at dimension " +
                                            std::to_string(i) + ": index value " +
                                            std::to_string(*(indexStart + i)) +
                                            " is >= shape dimension " + std::to_string(shape[i]));
                }
            }

            return baseTensor_->at(indexStart);
        }

        template <size_t N> inline void *at(const std::array<size_t, N> &index) const {
            return at(index.data(), N);
        }

        inline void *at(const std::vector<size_t> &index) const {
            return at(index.data(), index.size());
        }

        inline void *at(const std::initializer_list<size_t> &index) const {
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

        std::string toString() const {
            const std::array<size_t, internal::MAX_DIMS> &shape =
                baseTensor_->getLayout().getShape();
            const size_t ndim = baseTensor_->getLayout().getNDim();
            std::array<size_t, internal::MAX_DIMS> shapeProduct;
            size_t product = 1;
            for (int i = ndim - 1; i >= 0; i--) {
                product *= shape[i];
                shapeProduct[i] = product;
            }

            // TODO : Add padding
            std::string str = "";
            int idx = 0;
            for (auto it = begin(); it != end(); it++, idx++) {
                for (int i = 0; i < ndim; i++) {
                    if (idx % shapeProduct[i] == 0) {
                        if (idx != 0) {
                            str += std::string(ndim - i, ']');
                            str += ndim - i > 1 ? "\n\n" : "\n";
                        }
                        str += std::string(ndim - i, '[') + " ";
                        break;
                    }
                }

                // TODO : Deal with different datatypes
                str += std::to_string(*static_cast<float *>(*it)) + " ";
            }

            str += std::string(ndim, ']');

            return str;
        }
    };
} // namespace ty