#pragma once

#include "tityos/ty/backend/CUDABackend.h"
#include "tityos/ty/cuda/cuda_utils.h"
#include "tityos/ty/ops/cuda/CudaTensorView.h"
#include "tityos/ty/tensor/Tensor.h"

#include <cstdint>
#include <stdexcept>

namespace ty {
namespace internal {
    template <typename T>
    __global__ void contiguousCudaKernel(CudaTensorView<T> tensorView,
                                         CudaTensorView<T> resultTensorView,
                                         size_t numElements);

    template <typename T>
    void launchContiguousKernel(const BaseTensor& tensor, BaseTensor& result);
} // namespace internal
} // namespace ty