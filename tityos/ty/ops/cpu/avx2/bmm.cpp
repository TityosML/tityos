#include "tityos/ty/ops/cpu/avx2/bmm.h"

namespace ty {
namespace internal {
    template <typename T>
    void bmmAvx2Kernel(const TensorView<T>& outView,
                       const TensorView<T>& batch1View,
                       const TensorView<T>& batch2View) {
        using Vec = typename Avx2Traits<T>::Vec;
        constexpr int lanes = Avx2Traits<T>::lanes;
        constexpr size_t NACC = 4;

        const size_t B = outView.shape[0];
        const size_t M = outView.shape[1];
        const size_t N = batch1View.shape[2];
        const size_t K = outView.shape[2];
        const size_t total = B * M * K;

        T* outData = outView.data + outView.offset;
        const T* batch1Data = batch1View.data + batch1View.offset;
        const T* batch2Data = batch2View.data + batch2View.offset;

#pragma omp parallel for collapse(2) schedule(static)
        for (size_t batch = 0; batch < B; batch++) {
            for (size_t m = 0; m < M; m++) {

                size_t k = 0;
                for (; k + NACC * lanes <= K; k += NACC * lanes) {
                    size_t outIdx1 = batch * outView.strides[0] +
                                     m * outView.strides[1] +
                                     k * outView.strides[2];
                    size_t batch1Idx = batch * batch1View.strides[0] +
                                       m * batch1View.strides[1];
                    size_t batch2Idx = batch * batch2View.strides[0] +
                                       k * batch2View.strides[2];

                    Vec outVecs[NACC];
                    Vec batch2Vecs[NACC];

                    for (size_t i = 0; i < NACC; i++) {
                        outVecs[i] = Avx2Traits<T>::empty();
                    }

                    for (size_t n = 0; n < N; n++) {
                        Vec batch1Vec =
                            Avx2Traits<T>::set1(batch1Data[batch1Idx]);

                        for (size_t i = 0; i < NACC; i++) {
                            batch2Vecs[i] = Avx2Traits<T>::load(
                                batch2Data + batch2Idx + i * lanes);
                            outVecs[i] = Avx2Traits<T>::fma(
                                batch1Vec, batch2Vecs[i], outVecs[i]);
                        }

                        batch1Idx += batch1View.strides[2];
                        batch2Idx += batch2View.strides[1];
                    }

                    for (size_t i = 0; i < NACC; i++) {
                        Avx2Traits<T>::store(outData + outIdx1 + i * lanes,
                                             outVecs[i]);
                    }
                }

                for (; k < K; k++) {
                    size_t outIdx = batch * outView.strides[0] +
                                    m * outView.strides[1] +
                                    k * outView.strides[2];
                    outData[outIdx] = 0;

                    for (size_t n = 0; n < N; n++) {
                        size_t batch1Idx = batch * batch1View.strides[0] +
                                           m * batch1View.strides[1] +
                                           n * batch1View.strides[2];
                        size_t batch2Idx = batch * batch2View.strides[0] +
                                           n * batch2View.strides[1] +
                                           k * batch2View.strides[2];

                        outData[outIdx] +=
                            batch1Data[batch1Idx] * batch2Data[batch2Idx];
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
