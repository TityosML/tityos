#pragma once

#include "tityos/ty/export.h"
#include "tityos/ty/tensor/BaseTensor.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <cmath>
#include <cstring>
#include <iterator>
#include <sstream>
#include <string>
#include <tuple>

namespace ty {
TITYOS_EXPORT class Tensor {
  private:
    std::shared_ptr<internal::BaseTensor> baseTensor_;

  public:
    template <class DataContainer, class ShapeContainer,
              std::enable_if_t<
                  std::is_same_v<typename ShapeContainer::value_type, size_t> &&
                      std::is_trivially_copyable_v<
                          typename DataContainer::value_type>,
                  int> = 0>
    Tensor(const DataContainer& data, const ShapeContainer& shape,
           Device device = {DeviceType::CPU, 0}, DType dtype = DType::Float32) {
        using T = typename DataContainer::value_type;

        if (std::size(shape) > internal::MAX_DIMS) {
            throw std::runtime_error("Tensor shape has too many dimensions.");
        }

        size_t numElements = std::size(data);
        size_t numBytes = sizeof(T) * numElements;
        std::shared_ptr<internal::TensorStorage> dataStorage =
            std::make_shared<internal::TensorStorage>(numBytes, device);
        dataStorage->copyDataFromCpu(std::data(data), numBytes);

        std::array<size_t, internal::MAX_DIMS> storageShape{};
        int i = 0;
        for (size_t s : shape) {
            storageShape[i++] = s;
        }

        internal::ShapeStrides layout(storageShape, std::size(shape));

        baseTensor_ =
            std::make_shared<internal::BaseTensor>(dataStorage, layout, dtype);
    }

    template <typename T, class ShapeContainer>
    Tensor(std::initializer_list<T> data, const ShapeContainer& shape,
           Device device = {DeviceType::CPU, 0}, DType dtype = DType::Float32)
        : Tensor(std::vector<T>(data), shape, device, dtype) {}

    template <class DataContainer>
    Tensor(const DataContainer& data, std::initializer_list<size_t> shape,
           Device device = {DeviceType::CPU, 0}, DType dtype = DType::Float32)
        : Tensor(data, std::vector<size_t>(shape), device, dtype) {}

    template <typename T>
    Tensor(std::initializer_list<T> data, std::initializer_list<size_t> shape,
           Device device = {DeviceType::CPU, 0}, DType dtype = DType::Float32)
        : Tensor(std::vector<T>(data), std::vector<size_t>(shape), device,
                 dtype) {}

    explicit Tensor(std::shared_ptr<internal::BaseTensor> baseTensor)
        : baseTensor_(baseTensor) {}

    Tensor(const Tensor& other);

    Tensor(Tensor&& other) noexcept;

    Tensor& operator=(const Tensor& other);

    Tensor& operator=(Tensor&& other) noexcept;

    Tensor copy() const;

    Device getDevice() const;

    DType getDType() const;

    const std::array<size_t, internal::MAX_DIMS> getShape() const;

    const std::array<size_t, internal::MAX_DIMS> getStrides() const;

    size_t getSize() const;

    size_t getNDim() const;

    std::shared_ptr<internal::BaseTensor> getBaseTensor() const;

    void* at(const size_t* indexStart, const size_t indexSize) const;

    template <size_t N> void* at(const std::array<size_t, N>& index) const {
        return at(index.data(), N);
    }

    void* at(const std::vector<size_t>& index) const;

    void* at(const std::initializer_list<size_t>& index) const;

    template <typename T>
    T& elemAt(const size_t* indexStart, const size_t indexSize) {
        return *static_cast<T*>(at(indexStart, indexSize));
    }

    template <typename T, size_t N>
    T& elemAt(const std::array<size_t, N>& index) {
        return *static_cast<T*>(at(index));
    }

    template <typename T> T& elemAt(const std::vector<size_t>& index) {
        return *static_cast<T*>(at(index));
    }

    template <typename T>
    T& elemAt(const std::initializer_list<size_t>& index) {
        return *static_cast<T*>(at(index));
    }

    using Iterator = internal::BaseTensor::Iterator;

    Iterator begin();

    Iterator end();

    Iterator begin() const;

    Iterator end() const;

    std::string toString() const;

  private:
    std::string itemToStringCpu(const void* item, const DType dtype) const;
};
} // namespace ty