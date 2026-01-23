#include "tityos/ty/ops/cuda/add.h"

#include <cuda_runtime.h>

namespace ty {
namespace internal {
    template <typename T>
    __global__ void
    addCudaKernel(CudaTensorView<T> tensorView1, CudaTensorView<T> tensorView2,
                  CudaTensorView<T> resultTensorView, size_t numElements) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        // guard in case block is not full
        if (idx < numElements) {
            resultTensorView[idx] = tensorView1[idx] + tensorView2[idx];
        }
    };

    template <typename T>
    void launchAddKernel(const BaseTensor& tensor1, const BaseTensor& tensor2,
                         BaseTensor& result) {
        // convert all tensors to tensorviews
        CudaTensorView<T> tensorView1 = buildCudaTensorView<T>(tensor1);
        CudaTensorView<T> tensorView2 = buildCudaTensorView<T>(tensor2);
        CudaTensorView<T> resultTensorView = buildCudaTensorView<T>(result);

        size_t numElements = tensor1.getLayout().numElements();

        int threadsPerBlock = 256;
        int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        // launch the add kernel
        addCudaKernel<T><<<blocks, threadsPerBlock>>>(
            tensorView1, tensorView2, resultTensorView, numElements);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // force compiler to compile each type
    template void launchAddKernel<int8_t>(const BaseTensor& tensor1,
                                          const BaseTensor& tensor2,
                                          BaseTensor& result);
    template void launchAddKernel<uint8_t>(const BaseTensor& tensor1,
                                           const BaseTensor& tensor2,
                                           BaseTensor& result);
    template void launchAddKernel<int16_t>(const BaseTensor& tensor1,
                                           const BaseTensor& tensor2,
                                           BaseTensor& result);
    template void launchAddKernel<uint16_t>(const BaseTensor& tensor1,
                                            const BaseTensor& tensor2,
                                            BaseTensor& result);
    template void launchAddKernel<int32_t>(const BaseTensor& tensor1,
                                           const BaseTensor& tensor2,
                                           BaseTensor& result);
    template void launchAddKernel<uint32_t>(const BaseTensor& tensor1,
                                            const BaseTensor& tensor2,
                                            BaseTensor& result);
    template void launchAddKernel<int64_t>(const BaseTensor& tensor1,
                                           const BaseTensor& tensor2,
                                           BaseTensor& result);
    template void launchAddKernel<uint64_t>(const BaseTensor& tensor1,
                                            const BaseTensor& tensor2,
                                            BaseTensor& result);
    template void launchAddKernel<float>(const BaseTensor& tensor1,
                                         const BaseTensor& tensor2,
                                         BaseTensor& result);
    template void launchAddKernel<double>(const BaseTensor& tensor1,
                                          const BaseTensor& tensor2,
                                          BaseTensor& result);

} // namespace internal
} // namespace ty
