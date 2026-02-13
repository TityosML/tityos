#pragma once

#include "tityos/ty/export.h"
#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/Index.h"
#include "tityos/ty/tensor/Indexing.h"
#include "tityos/ty/tensor/ShapeStrides.h"
#include "tityos/ty/tensor/TensorStorage.h"

#include <array>
#include <memory>
#include <vector>

namespace ty {
namespace internal {
    class TITYOS_API BaseTensor {
      private:
        std::shared_ptr<TensorStorage> tensorStorage_;
        ShapeStrides layout_;
        DType dtype_;

      public:
        BaseTensor(std::shared_ptr<TensorStorage> data,
                   const ShapeStrides& layout,
                   const DType dtype = DType::Float32);
        BaseTensor(const BaseTensor& other);
        BaseTensor(BaseTensor&& other) noexcept;

        BaseTensor& operator=(const BaseTensor& other);
        BaseTensor& operator=(BaseTensor&& other) noexcept;

        BaseTensor copy() const;

        void* at(const size_t* indexStart) const;
        void* at(size_t index) const;

        template <typename T> T* elemAt(const size_t* indexStart) const {
            return reinterpret_cast<T*>(at(indexStart));
        }

        template <typename T> T* elemAt(size_t index) const {
            return reinterpret_cast<T*>(at(index));
        }

        const ShapeStrides& getLayout() const;
        size_t getNDim() const;
        const TensorShape& getShape() const;
        const TensorStrides& getStrides() const;
        size_t getLogicalSize() const;
        size_t getSize() const;
        const std::shared_ptr<TensorStorage>& getTensorStorage() const;
        Device getDevice() const;
        DType getDType() const;

        bool isContiguous() const;

        bool operator==(const BaseTensor& other) const;

        BaseTensor slice(size_t dim,
                         std::optional<ptrdiff_t> start = std::nullopt,
                         std::optional<ptrdiff_t> stop = std::nullopt,
                         ptrdiff_t step = 1) const;

        BaseTensor indexList(IndexList indices) const;

        struct Iterator {
          private:
            const BaseTensor& baseTensor_;
            size_t linearIndex_;
            void* ptr_;

          public:
            Iterator(const BaseTensor& baseTensor, size_t linearStartIndex);

            size_t getIndex() const;

            void jumpToIndex(size_t linearIndex);

            void* operator->();
            void* operator*();

            Iterator& operator++();
            Iterator operator++(int);

            TITYOS_API friend bool operator==(const Iterator& a,
                                              const Iterator& b);
            TITYOS_API friend bool operator!=(const Iterator& a,
                                              const Iterator& b);
        };

        Iterator begin();
        Iterator end();

        Iterator begin() const;
        Iterator end() const;

      private:
        size_t endIndex() const;
    };
} // namespace internal
} // namespace ty