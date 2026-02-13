#pragma once

#include "tityos/ty/Device.h"
#include "tityos/ty/backend/Backend.h"
#include "tityos/ty/export.h"
#include "tityos/ty/tensor/Dtype.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ty {
namespace internal {
    class TITYOS_API TensorStorage {
      private:
        void* startPointer_;
        size_t size_;
        Device device_;

      public:
        TensorStorage(size_t numBytes, Device device = {DeviceType::CPU, 0});
        ~TensorStorage();

        TensorStorage(const TensorStorage& other);
        TensorStorage(TensorStorage&& other) noexcept;
        TensorStorage& operator=(const TensorStorage& other);
        TensorStorage& operator=(TensorStorage&& other) noexcept;

        void copyDataFromCpu(const void* dataStartPointer, size_t numBytes);

        size_t getSize() const;
        Device getDevice() const;

        void* at(size_t index);
        void* begin();
        void* end();

      private:
        void allocate();
        void deallocate();
    };
} // namespace internal
} // namespace ty