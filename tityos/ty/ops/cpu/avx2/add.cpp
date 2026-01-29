#include "tityos/ty/ops/cpu/avx2/add.h"

namespace ty {
namespace internal {
    template <typename T>
    void addAvx2Kernel(const TensorView<T>& outView,
                       const TensorView<T>& tensor1View,
                       const TensorView<T>& tensor2View) {
        using Vec = typename Avx2Traits<T>::Vec;
        constexpr int lanes = Avx2Traits<T>::lanes;

        const size_t total =
            std::accumulate(outView.shape, outView.shape + outView.ndim, 1,
                            std::multiplies<size_t>());

        T* outData = outView.data + outView.offset;
        const T* tensor1Data = tensor1View.data + tensor1View.offset;
        const T* tensor2Data = tensor2View.data + tensor2View.offset;

        size_t i = 0;
        for (; i + lanes <= total; i += lanes) {
            using Vec = typename Avx2Traits<T>::Vec;
            Vec va = Avx2Traits<T>::load(tensor1Data + i);
            Vec vb = Avx2Traits<T>::load(tensor2Data + i);
            Vec vc = Avx2Traits<T>::add(va, vb);
            Avx2Traits<T>::store(outData + i, vc);
        }

        for (; i < total; i++) {
            outData[i] = tensor1Data[i] + tensor2Data[i];
        }
    }

    void addAvx2(BaseTensor& result, const BaseTensor& tensor1,
                       const BaseTensor& tensor2) {
        switch (result.getDType()) {
        case DType::Int8:
            addAvx2Kernel<int8_t>(result, tensor1, tensor2);
            break;
        case DType::UInt8:
            addAvx2Kernel<uint8_t>(result, tensor1, tensor2);
            break;
        case DType::Int16:
            addAvx2Kernel<int16_t>(result, tensor1, tensor2);
            break;
        case DType::UInt16:
            addAvx2Kernel<uint16_t>(result, tensor1, tensor2);
            break;
        case DType::Int32:
            addAvx2Kernel<int32_t>(result, tensor1, tensor2);
            break;
        case DType::UInt32:
            addAvx2Kernel<uint32_t>(result, tensor1, tensor2);
            break;
        case DType::Int64:
            addAvx2Kernel<int64_t>(result, tensor1, tensor2);
            break;
        case DType::UInt64:
            addAvx2Kernel<uint64_t>(result, tensor1, tensor2);
            break;
        case DType::Float32:
            addAvx2Kernel<float>(result, tensor1, tensor2);
            break;
        case DType::Float64:
            addAvx2Kernel<double>(result, tensor1, tensor2);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for addition");
        }
    }
} // namespace internal
} // namespace ty
