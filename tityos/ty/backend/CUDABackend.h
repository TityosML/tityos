#pragma once

#include "tityos/ty/backend/Backend.h"

namespace ty {
namespace internal {
    namespace backend {
        class CUDABackend final : public Backend {
          public:
            void* allocate(size_t bytes, int index) override;
            void deallocate(void* ptr) override;
            void copyData(void* destPtr, const void* srcPtr,
                          size_t numBytes) override;
            void copyDataFromCpu(void* destPtr, const void* srcPtr,
                                 size_t numBytes) override;

            BaseTensor add(const BaseTensor& tensor1,
                           const BaseTensor& tensor2) override;
            BaseTensor bmm(const BaseTensor& batch1,
                           const BaseTensor& batch2) override;
        };
    } // namespace backend
} // namespace internal
} // namespace ty

extern "C" ty::internal::backend::Backend* registerCudaBackend();