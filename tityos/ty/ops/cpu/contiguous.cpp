#include "tityos/ty/ops/cpu/contiguous.h"

namespace ty {
namespace internal {
    template <typename T>
    void contiguousCpuKernel(const TensorView& outView,
                             const TensorView& tensorView) {
        size_t total =
            std::accumulate(outView.shape, outView.shape + outView.ndim, 1,
                            std::multiplies<size_t>());

        T* __restrict__ outData =
            static_cast<T*>(outView.data) + outView.offset;
        const T* __restrict__ tensorData =
            static_cast<T*>(tensorView.data) + tensorView.offset;

        size_t idx[TY_MAX_DIMS];
        for (size_t i = 0; i < outView.ndim; i++) {
            idx[i] = 0;
        }

        for (size_t linear = 0; linear < total; linear++) {
            ptrdiff_t outIdx = 0;
            ptrdiff_t tensorIdx = 0;

            for (size_t i = 0; i < outView.ndim; i++) {
                outIdx += idx[i] * outView.strides[i];
                tensorIdx += idx[i] * tensorView.strides[i];
            }

            outData[outIdx] = tensorData[tensorIdx];

            for (size_t i = outView.ndim; i-- > 0;) {
                if (++idx[i] < outView.shape[i]) {
                    break;
                }

                idx[i] = 0;
            }
        }
    }

    BaseTensor backend::CPUBackend::contiguous(const BaseTensor& tensor) {
        BaseTensor result = internal::emptyLike(tensor);

        DISPATCH_KERNEL_DTYPE_TABLE(kernelTable, contiguousCpuKernel,
                                    (const TensorView&, const TensorView&))

        kernelTable[static_cast<size_t>(tensor.getDType())](result, tensor);

        return result;
    }
}; // namespace internal
} // namespace ty