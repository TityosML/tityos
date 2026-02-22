#include "tityos/ty/ops/cpu/add.h"

#include <iostream>

namespace ty {
namespace internal {
    template <typename T>
    void addCpuKernel(const TensorView& outView, const TensorView& tensor1View, const TensorView& tensor2View) {
        size_t total = std::accumulate(outView.shape, outView.shape + outView.ndim, 1, std::multiplies<size_t>());

        T* __restrict__ outData = static_cast<T*>(outView.data) + outView.offset;
        const T* __restrict__ tensor1Data = static_cast<T*>(tensor1View.data) + tensor1View.offset;
        const T* __restrict__ tensor2Data = static_cast<T*>(tensor2View.data) + tensor2View.offset;

        size_t idx[TY_MAX_DIMS];
        for (size_t i = 0; i < outView.ndim; i++) {
            idx[i] = 0;
        }

        for (size_t linear = 0; linear < total; linear++) {
            ptrdiff_t outIdx = 0;
            ptrdiff_t tensor1Idx = 0;
            ptrdiff_t tensor2Idx = 0;

            for (size_t i = 0; i < outView.ndim; i++) {
                outIdx += idx[i] * outView.strides[i];
                tensor1Idx += idx[i] * tensor1View.strides[i];
                tensor2Idx += idx[i] * tensor2View.strides[i];
            }

            outData[outIdx] = tensor1Data[tensor1Idx] + tensor2Data[tensor2Idx];

            for (size_t i = outView.ndim; i-- > 0;) {
                if (++idx[i] < outView.shape[i]) {
                    break;
                }

                idx[i] = 0;
            }
        }
    }

    BaseTensor backend::CPUBackend::add(const BaseTensor& tensor1, const BaseTensor& tensor2) {
        BaseTensor result = internal::emptyLike(tensor1);

        // Avx Optimized kernel
        if (tensor1.isContiguous() && tensor2.isContiguous()) {
            addAvx(result, tensor1, tensor2);
            return result;
        }

        DISPATCH_KERNEL_DTYPE_TABLE(kernelTable, addCpuKernel,
                                    (const TensorView&, const TensorView&, const TensorView&))

        kernelTable[static_cast<size_t>(tensor1.getDType())](result, tensor1, tensor2);

        return result;
    }
} // namespace internal
} // namespace ty
