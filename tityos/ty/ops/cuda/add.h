#pragma once

#include "tityos/ty/backend/CUDABackend.h"
#include "tityos/ty/cuda/cuda_utils.h"
#include "tityos/ty/ops/cuda/CudaTensorView.h"
#include "tityos/ty/ops/dispatchDType.h"
#include "tityos/ty/ops/empty.h"
#include "tityos/ty/tensor/BaseTensor.h"

#include <cstdint>
#include <stdexcept>

namespace ty {
namespace internal {
    template <typename T>
    __global__ void
    addCudaKernel(CudaTensorView<T> tensorView1, CudaTensorView<T> tensorView2,
                  CudaTensorView<T> resultTensorView, size_t numElements);

    template <typename T>
    void launchAddKernel(BaseTensor& result, const BaseTensor& tensor1,
                         const BaseTensor& tensor2);
} // namespace internal
} // namespace ty