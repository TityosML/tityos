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
            resultTensorView[idx] = tensorView.atLinear(idx);
        }
    };

    template <typename T>
    void launchContiguousKernel(BaseTensor& result, const BaseTensor& tensor) {

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
    template void launchContiguousKernel<int8_t>(BaseTensor& result,
                                                 const BaseTensor& tensor);
    template void launchContiguousKernel<uint8_t>(BaseTensor& result,
                                                  const BaseTensor& tensor);
    template void launchContiguousKernel<int16_t>(BaseTensor& result,
                                                  const BaseTensor& tensor);
    template void launchContiguousKernel<uint16_t>(BaseTensor& result,
                                                   const BaseTensor& tensor);
    template void launchContiguousKernel<int32_t>(BaseTensor& result,
                                                  const BaseTensor& tensor);
    template void launchContiguousKernel<uint32_t>(BaseTensor& result,
                                                   const BaseTensor& tensor);
    template void launchContiguousKernel<int64_t>(BaseTensor& result,
                                                  const BaseTensor& tensor);
    template void launchContiguousKernel<uint64_t>(BaseTensor& result,
                                                   const BaseTensor& tensor);
    template void launchContiguousKernel<float>(BaseTensor& result,
                                                const BaseTensor& tensor);
    template void launchContiguousKernel<double>(BaseTensor& result,
                                                 const BaseTensor& tensor);

} // namespace internal
} // namespace ty
