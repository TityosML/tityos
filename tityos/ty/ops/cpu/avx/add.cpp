#include "tityos/ty/ops/cpu/avx/add.h"

namespace ty {
namespace internal {
    template <typename T>
    void addAvxKernel(const TensorView& outView, const TensorView& tensor1View,
                      const TensorView& tensor2View) {
        using Vec = typename AvxTraits<T>::Vec;
        constexpr int lanes = AvxTraits<T>::lanes;

        const size_t total =
            std::accumulate(outView.shape, outView.shape + outView.ndim, 1,
                            std::multiplies<size_t>());

        T* outData = static_cast<T*>(outView.data) + outView.offset;
        const T* tensor1Data =
            static_cast<T*>(tensor1View.data) + tensor1View.offset;
        const T* tensor2Data =
            static_cast<T*>(tensor2View.data) + tensor2View.offset;

        size_t i = 0;
        for (; i + lanes <= total; i += lanes) {
            Vec va = AvxTraits<T>::load(tensor1Data + i);
            Vec vb = AvxTraits<T>::load(tensor2Data + i);
            Vec vc = AvxTraits<T>::add(va, vb);
            AvxTraits<T>::store(outData + i, vc);
        }

        for (; i < total; i++) {
            outData[i] = tensor1Data[i] + tensor2Data[i];
        }
    }

    void addAvx(BaseTensor& result, const BaseTensor& tensor1,
                const BaseTensor& tensor2) {
        DISPATCH_KERNEL_DTYPE_TABLE(
            kernelTable, addAvxKernel,
            (const TensorView&, const TensorView&, const TensorView&))

        kernelTable[static_cast<size_t>(tensor1.getDType())](result, tensor1,
                                                             tensor2);
    }
} // namespace internal
} // namespace ty
