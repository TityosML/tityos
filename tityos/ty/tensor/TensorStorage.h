#pragma once

#include <cstdlib>
#include <stdexcept>

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
                return reinterpret_cast<char *>(startPointer_) + index;
            }

            size_t getSize() const {
                return size_;
            }

            void *begin() {
                return startPointer_;
            }

            void *end() {
                return reinterpret_cast<char *>(startPointer_) + size_;
            }

          private:
            void allocate();
            void deallocate();
        };
    } // namespace internal
} // namespace ty