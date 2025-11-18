#pragma once

#include <cstdlib>

#include "tityos/ty/cuda/cuda_import.h"
#include "tityos/ty/cuda/cuda_utils.h"
#include "tityos/ty/tensor/Device.h"
#include "tityos/ty/tensor/Dtype.h"

namespace ty {
    namespace internal {
        class TensorStorage {
          private:
            void *startPointer_;
            size_t size_;
            Device device_;

          public:
            TensorStorage(size_t numBytes, Device device = {DeviceType::CPU, 0})
                : size_(numBytes), device_(device) {
                allocate();
            }

            ~TensorStorage() {
                deallocate();
            }

            void *at(size_t index) {
                return static_cast<char *>(startPointer_) + index;
            }

            size_t getSize() const {
                return size_;
            }

          private:
            void allocate();
            void deallocate();
        };
    } // namespace internal
} // namespace ty