#include "tityos/ty/ops/cpu/bmm.h"

namespace ty {
namespace internal {
    template <typename T>
    void bmmCpuKernel(const TensorView& outView, const TensorView& tensor1View,
                      const TensorView& tensor2View) {
        T* __restrict__ outData =
            static_cast<T*>(outView.data) + outView.offset;
        const T* __restrict__ tensor1Data =
            static_cast<T*>(tensor1View.data) + tensor1View.offset;
        const T* __restrict__ tensor2Data =
            static_cast<T*>(tensor2View.data) + tensor2View.offset;

#pragma omp parallel for
        for (size_t batch = 0; batch < outView.shape[0]; batch++) {
            for (size_t m = 0; m < tensor1View.shape[1]; m++) {
                for (size_t k = 0; k < tensor2View.shape[2]; k++) {
                    size_t outIdx = batch * outView.strides[0] +
                                    m * outView.strides[1] +
                                    k * outView.strides[2];

                    outData[outIdx] = 0;
                    for (size_t n = 0; n < tensor2View.shape[1]; n++) {
                        size_t tensor1Idx = batch * tensor1View.strides[0] +
                                            m * tensor1View.strides[1] +
                                            n * tensor1View.strides[2];
                        size_t tensor2Idx = batch * tensor2View.strides[0] +
                                            n * tensor2View.strides[1] +
                                            k * tensor2View.strides[2];

                        outData[outIdx] +=
                            tensor1Data[tensor1Idx] * tensor2Data[tensor2Idx];
                    }
                }
            }
        }
    }

    BaseTensor backend::CPUBackend::bmm(const BaseTensor& batch1,
                                        const BaseTensor& batch2) {
        auto shape1 = batch1.getShape();
        auto shape2 = batch2.getShape();

        TensorShape resultShape = {shape1[0], shape1[1], shape2[2]};
        BaseTensor result = internal::empty(resultShape, 3, batch1.getDType(),
                                            batch1.getDevice());

        // Avx Optimized kernel
        if (batch1.isContiguous() && batch2.isContiguous()) {
            bmmAvx(result, batch1, batch2);
            return result;
        }

        DISPATCH_KERNEL_DTYPE_TABLE(
            kernelTable, bmmCpuKernel,
            (const TensorView&, const TensorView&, const TensorView&))

        kernelTable[static_cast<size_t>(batch1.getDType())](result, batch1,
                                                             batch2);

        return result;
    }
}; // namespace internal
} // namespace ty