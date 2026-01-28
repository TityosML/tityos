#include "tityos/ty/ops/cuda/contiguous.h"

#include <cuda_runtime.h>

namespace ty {
namespace internal {
    template <typename T>
    __global__ void contiguousCudaKernel(CudaTensorView<T> tensorView,
                                         CudaTensorView<T> resultTensorView,
                                         size_t numElements) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        // result tensor view is assumed contiguous allowing []
        if (idx < numElements) {
            resultTensorView.atLinear(idx) = tensorView.atLinear(idx);
        }
    };

    template <typename T>
    void launchContiguousKernel(const BaseTensor& tensor, BaseTensor& result) {

        // convert all tensors to tensorviews
        CudaTensorView<T> tensorView = buildCudaTensorView<T>(tensor);
        CudaTensorView<T> resultTensorView = buildCudaTensorView<T>(result);

        size_t numElements = tensor.getLayout().numElements();

        int threadsPerBlock = 256;
        int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        // launch the add kernel
        contiguousCudaKernel<T><<<blocks, threadsPerBlock>>>(
            tensorView, resultTensorView, numElements);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // force compiler to compile each type
    template void launchContiguousKernel<int8_t>(const BaseTensor& tensor,
                                                 BaseTensor& result);
    template void launchContiguousKernel<uint8_t>(const BaseTensor& tensor,
                                                  BaseTensor& result);
    template void launchContiguousKernel<int16_t>(const BaseTensor& tensor,
                                                  BaseTensor& result);
    template void launchContiguousKernel<uint16_t>(const BaseTensor& tensor,
                                                   BaseTensor& result);
    template void launchContiguousKernel<int32_t>(const BaseTensor& tensor,
                                                  BaseTensor& result);
    template void launchContiguousKernel<uint32_t>(const BaseTensor& tensor,
                                                   BaseTensor& result);
    template void launchContiguousKernel<int64_t>(const BaseTensor& tensor,
                                                  BaseTensor& result);
    template void launchContiguousKernel<uint64_t>(const BaseTensor& tensor,
                                                   BaseTensor& result);
    template void launchContiguousKernel<float>(const BaseTensor& tensor,
                                                BaseTensor& result);
    template void launchContiguousKernel<double>(const BaseTensor& tensor,
                                                 BaseTensor& result);

} // namespace internal
} // namespace ty
