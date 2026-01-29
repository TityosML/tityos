#include "tityos/ty/ops/cpu/avx2/bmm.h"

namespace ty {
namespace internal {
    template <typename T>
    void bmmAvx2Kernel(const TensorView<T>& outView,
                       const TensorView<T>& batch1View,
                       const TensorView<T>& batch2View) {
        using Vec = typename Avx2Traits<T>::Vec;
        constexpr int lanes = Avx2Traits<T>::lanes;

        T* outData = outView.data + outView.offset;
        const T* batch1Data = batch1View.data + batch1View.offset;
        const T* batch2Data = batch2View.data + batch2View.offset;

        T buffer[lanes];

        size_t n;
        size_t outIdx;
        size_t batch1Idx;
        size_t batch2Idx;

        for (size_t batch = 0; batch < outView.shape[0]; batch++) {

            for (size_t k = 0; k < batch2View.shape[2]; k++) {

                // Grouping values
                n = 0;
                for (; n + lanes <= batch1View.shape[2]; n += lanes) {
                    batch2Idx = batch * batch2View.strides[0] +
                                n * batch2View.strides[1] +
                                k * batch2View.strides[2];

                    // Load the partial rows from batch2
                    for (size_t i = 0; i < lanes; i++) {
                        buffer[i] = batch2Data[batch2Idx];
                        batch2Idx += batch2View.strides[2];
                    }
                    Vec vecBatch2 = Avx2Traits<T>::load(buffer);

                    for (size_t m = 0; m < batch1View.shape[1]; m++) {
                        outIdx = batch * outView.strides[0] +
                                 m * outView.strides[1] +
                                 k * outView.strides[2];
                        batch1Idx = batch * batch1View.strides[0] +
                                    m * batch1View.strides[1] +
                                    n * batch1View.strides[2];

                        Vec vecBatch1 =
                            Avx2Traits<T>::load(batch1Data + batch1Idx);
                        Vec probVec = Avx2Traits<T>::mul(vecBatch1, vecBatch2);

                        if (n == 0) {
                            outData[outIdx] = 0;
                        }

                        outData[outIdx] += Avx2Traits<T>::sum(probVec);
                    }
                }

                // Trailing values
                for (; n < batch1View.shape[2]; n++) {
                    batch2Idx = batch * batch2View.strides[0] +
                                n * batch2View.strides[1] +
                                k * batch2View.strides[2];
                    for (size_t m = 0; m < batch1View.shape[1]; m++) {
                        outIdx = batch * outView.strides[0] +
                                 m * outView.strides[1] +
                                 k * outView.strides[2];
                        batch1Idx = batch * batch1View.strides[0] +
                                    m * batch1View.strides[1] +
                                    n * batch1View.strides[2];

                        if (n == 0) {
                            outData[outIdx] = 0;
                        }

                        outData[outIdx] += batch1Data[batch1Idx] * batch2Data[batch2Idx];
                    }
                }
            }
        }
    }

    void bmmAvx2(BaseTensor& result, const BaseTensor& batch1,
                 const BaseTensor& batch2) {
        switch (result.getDType()) {
        case DType::Int8:
            bmmAvx2Kernel<int8_t>(result, batch1, batch2);
            break;
        case DType::UInt8:
            bmmAvx2Kernel<uint8_t>(result, batch1, batch2);
            break;
        case DType::Int16:
            bmmAvx2Kernel<int16_t>(result, batch1, batch2);
            break;
        case DType::UInt16:
            bmmAvx2Kernel<uint16_t>(result, batch1, batch2);
            break;
        case DType::Int32:
            bmmAvx2Kernel<int32_t>(result, batch1, batch2);
            break;
        case DType::UInt32:
            bmmAvx2Kernel<uint32_t>(result, batch1, batch2);
            break;
        case DType::Int64:
            bmmAvx2Kernel<int64_t>(result, batch1, batch2);
            break;
        case DType::UInt64:
            bmmAvx2Kernel<uint64_t>(result, batch1, batch2);
            break;
        case DType::Float32:
            bmmAvx2Kernel<float>(result, batch1, batch2);
            break;
        case DType::Float64:
            bmmAvx2Kernel<double>(result, batch1, batch2);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for addition");
        }
    }
} // namespace internal
} // namespace ty
