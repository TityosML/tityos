#include "tityos/ty/ops/cuda/add.h"

#include <cuda_runtime.h>

namespace ty {
namespace internal {
    template <typename T>
    __global__ void addCudaKernel(CudaTensorView<T> tensorView1, CudaTensorView<T> tensorView2,
                                  CudaTensorView<T> resultTensorView, size_t numElements) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        // guard in case block is not full
        if (idx < numElements) {
            resultTensorView.atLinear(idx) = tensorView1.atLinear(idx) + tensorView2.atLinear(idx);
        }
    };

    template <typename T>
    void launchAddKernel(BaseTensor& result, const BaseTensor& tensor1, const BaseTensor& tensor2) {
        // convert all tensors to tensorviews
        CudaTensorView<T> tensorView1 = buildCudaTensorView<T>(tensor1);
        CudaTensorView<T> tensorView2 = buildCudaTensorView<T>(tensor2);
        CudaTensorView<T> resultTensorView = buildCudaTensorView<T>(result);

        size_t numElements = tensor1.getLayout().numElements();

        int threadsPerBlock = 256;
        int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        // launch the add kernel
        addCudaKernel<T><<<blocks, threadsPerBlock>>>(tensorView1, tensorView2, resultTensorView, numElements);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // force compiler to compile each type
    template void launchAddKernel<int8_t>(BaseTensor& result, const BaseTensor& tensor1, const BaseTensor& tensor2);
    template void launchAddKernel<uint8_t>(BaseTensor& result, const BaseTensor& tensor1, const BaseTensor& tensor2);
    template void launchAddKernel<int16_t>(BaseTensor& result, const BaseTensor& tensor1, const BaseTensor& tensor2);
    template void launchAddKernel<uint16_t>(BaseTensor& result, const BaseTensor& tensor1, const BaseTensor& tensor2);
    template void launchAddKernel<int32_t>(BaseTensor& result, const BaseTensor& tensor1, const BaseTensor& tensor2);
    template void launchAddKernel<uint32_t>(BaseTensor& result, const BaseTensor& tensor1, const BaseTensor& tensor2);
    template void launchAddKernel<int64_t>(BaseTensor& result, const BaseTensor& tensor1, const BaseTensor& tensor2);
    template void launchAddKernel<uint64_t>(BaseTensor& result, const BaseTensor& tensor1, const BaseTensor& tensor2);
    template void launchAddKernel<float>(BaseTensor& result, const BaseTensor& tensor1, const BaseTensor& tensor2);
    template void launchAddKernel<double>(BaseTensor& result, const BaseTensor& tensor1, const BaseTensor& tensor2);

} // namespace internal
} // namespace ty
