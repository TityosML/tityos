#include "tityos/ty/ops/cpu/avx/add.h"

namespace ty {
namespace internal {
    template <typename T>
    void addAvxKernel(const TensorView<T>& outView,
                       const TensorView<T>& tensor1View,
                       const TensorView<T>& tensor2View) {
        using Vec = typename AvxTraits<T>::Vec;
        constexpr int lanes = AvxTraits<T>::lanes;

        const size_t total =
            std::accumulate(outView.shape, outView.shape + outView.ndim, 1,
                            std::multiplies<size_t>());

        T* outData = outView.data + outView.offset;
        const T* tensor1Data = tensor1View.data + tensor1View.offset;
        const T* tensor2Data = tensor2View.data + tensor2View.offset;

        size_t i = 0;
        for (; i + lanes <= total; i += lanes) {
            using Vec = typename AvxTraits<T>::Vec;
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
        switch (result.getDType()) {
        case DType::Int8:
            addAvxKernel<int8_t>(result, tensor1, tensor2);
            break;
        case DType::UInt8:
            addAvxKernel<uint8_t>(result, tensor1, tensor2);
            break;
        case DType::Int16:
            addAvxKernel<int16_t>(result, tensor1, tensor2);
            break;
        case DType::UInt16:
            addAvxKernel<uint16_t>(result, tensor1, tensor2);
            break;
        case DType::Int32:
            addAvxKernel<int32_t>(result, tensor1, tensor2);
            break;
        case DType::UInt32:
            addAvxKernel<uint32_t>(result, tensor1, tensor2);
            break;
        case DType::Int64:
            addAvxKernel<int64_t>(result, tensor1, tensor2);
            break;
        case DType::UInt64:
            addAvxKernel<uint64_t>(result, tensor1, tensor2);
            break;
        case DType::Float32:
            addAvxKernel<float>(result, tensor1, tensor2);
            break;
        case DType::Float64:
            addAvxKernel<double>(result, tensor1, tensor2);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for addition");
        }
    }
} // namespace internal
} // namespace ty
