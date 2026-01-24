#include "tityos/ty/ops/cuda/bmm.h"

#include <cuda_runtime.h>

namespace ty {
namespace internal {
    template <typename T>
    __global__ void
    bmmCudaKernel(CudaTensorView<T> tensorView1, CudaTensorView<T> tensorView2,
                  CudaTensorView<T> resultTensorView, size_t numElements) {
        // TODO: Implement bmmCudaKernel
    };

    template <typename T>
    void launchBMMKernel(const BaseTensor& batch1, const BaseTensor& batch2,
                         BaseTensor& result) {
        // TODO: Implement launchBMMKernel
    }

    // force compiler to compile each type
    template void launchBMMKernel<int8_t>(const BaseTensor& batch1,
                                          const BaseTensor& batch2,
                                          BaseTensor& result);
    template void launchBMMKernel<uint8_t>(const BaseTensor& batch1,
                                          const BaseTensor& batch2,
                                          BaseTensor& result);
    template void launchBMMKernel<int16_t>(const BaseTensor& batch1,
                                          const BaseTensor& batch2,
                                          BaseTensor& result);
    template void launchBMMKernel<uint16_t>(const BaseTensor& batch1,
                                          const BaseTensor& batch2,
                                          BaseTensor& result);
    template void launchBMMKernel<int32_t>(const BaseTensor& batch1,
                                          const BaseTensor& batch2,
                                          BaseTensor& result);
    template void launchBMMKernel<uint32_t>(const BaseTensor& batch1,
                                          const BaseTensor& batch2,
                                          BaseTensor& result);
    template void launchBMMKernel<int64_t>(const BaseTensor& batch1,
                                          const BaseTensor& batch2,
                                          BaseTensor& result);
    template void launchBMMKernel<uint64_t>(const BaseTensor& batch1,
                                          const BaseTensor& batch2,
                                          BaseTensor& result);
    template void launchBMMKernel<float>(const BaseTensor& batch1,
                                          const BaseTensor& batch2,
                                          BaseTensor& result);
    template void launchBMMKernel<double>(const BaseTensor& batch1,
                                          const BaseTensor& batch2,
                                          BaseTensor& result);

} // namespace internal
} // namespace ty
