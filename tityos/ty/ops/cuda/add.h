#pragma once

#include "tityos/ty/ops/CudaTensorView.h"
#include "tityos/ty/tensor/Tensor.h"

#include <cstdint>
#include <stdexcept>

namespace ty {
namespace internal {
    BaseTensor internalAddCuda(const BaseTensor& tensor1,
                               const BaseTensor& tensor2);

    template <typename T>
    __global__ void
    addCudaKernel(CudaTensorView<T> tensorView1, CudaTensorView<T> tensorView2,
                  CudaTensorView<T> resultTensorView, size_t numElements);

    template <typename T>
    void launchAddKernel(const BaseTensor& tensor1, const BaseTensor& tensor2,
                         BaseTensor& result);
} // namespace internal
} // namespace ty