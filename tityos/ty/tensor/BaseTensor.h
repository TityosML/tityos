#pragma once

#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"
#include "tityos/ty/tensor/TensorStorage.h"

#include <array>
#include <memory>
#include <vector>

namespace ty {
namespace internal {
class BaseTensor {
  private:
    DType dtype_;

    std::shared_ptr<TensorStorage> tensorStorage_;
    ShapeStrides layout_;

    const std::array<size_t, MAX_DIMS> endIndex() const;

  public:
    BaseTensor(std::shared_ptr<TensorStorage> data, const ShapeStrides& layout,
               const DType dtype = DType::Float32);
    BaseTensor(const BaseTensor& other);
    BaseTensor(BaseTensor&& other) noexcept;

    BaseTensor& operator=(const BaseTensor& other);
    BaseTensor& operator=(BaseTensor&& other) noexcept;

    void* at(const size_t* indexStart) const;

    const ShapeStrides& getLayout() const;

    const std::shared_ptr<TensorStorage>& getTensorStorage() const;

    const DType getDType() const;

    struct Iterator {
      private:
        const BaseTensor& baseTensor_;

        std::array<size_t, MAX_DIMS> index_;

        void* ptr_;

        const std::array<size_t, MAX_DIMS> endIndex() const;

        void incrementIndex();

      public:
        Iterator(const BaseTensor& baseTensor,
                 const std::array<size_t, MAX_DIMS> startIndex);

        void* operator->();
        void* operator*();

        Iterator& operator++();

        Iterator operator++(int);

        friend bool operator==(const Iterator& a, const Iterator& b);

        friend bool operator!=(const Iterator& a, const Iterator& b);
    };

    Iterator begin();
    Iterator end();

    Iterator begin() const;
    Iterator end() const;
};
} // namespace internal
} // namespace ty