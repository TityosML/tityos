#include "tityos/ty/ops/cpu/avx2/bmm.h"

namespace ty {
namespace internal {
    template <typename T>
    void bmmAvx2Kernel(const TensorView<T>& outView,
                       const TensorView<T>& batch1View,
                       const TensorView<T>& batch2View) {
        using Vec = typename Avx2Traits<T>::Vec;
        constexpr int lanes = Avx2Traits<T>::lanes;
        constexpr size_t NACC = 2;

        const size_t B = outView.shape[0];
        const size_t M = outView.shape[1];
        const size_t N = batch1View.shape[2];
        const size_t K = outView.shape[2];
        const size_t total = B * M * K;

        T* outData = outView.data + outView.offset;
        const T* batch1Data = batch1View.data + batch1View.offset;
        const T* batch2Data = batch2View.data + batch2View.offset;

        const auto outStride0 = outView.strides[0];
        const auto outStride1 = outView.strides[1];
        const auto outStride2 = outView.strides[2];

        const auto batch1Stride0 = batch1View.strides[0];
        const auto batch1Stride1 = batch1View.strides[1];
        const auto batch1Stride2 = batch1View.strides[2];

        const auto batch2Stride0 = batch2View.strides[0];
        const auto batch2Stride1 = batch2View.strides[1];
        const auto batch2Stride2 = batch2View.strides[2];

#pragma omp parallel for collapse(2) schedule(guided)
        for (size_t batch = 0; batch < B; batch++) {
            for (size_t m = 0; m < M; m++) {
                Vec outVecs[NACC];
                Vec batch2Vecs[NACC];

                size_t k = 0;
                for (; k + NACC * lanes <= K; k += NACC * lanes) {
                    size_t outIdx =
                        batch * outStride0 + m * outStride1 + k * outStride2;
                    size_t batch1Idx =
                        batch * batch1Stride0 + m * batch1Stride1;
                    size_t batch2Idx =
                        batch * batch2Stride0 + k * batch2Stride2;

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

                        batch1Idx += batch1Stride2;
                        batch2Idx += batch2Stride1;
                    }

                    for (size_t i = 0; i < NACC; i++) {
                        Avx2Traits<T>::store(outData + outIdx + i * lanes,
                                             outVecs[i]);
                    }
                }

                for (; k + lanes <= K; k += lanes) {
                    size_t outIdx =
                        batch * outStride0 + m * outStride1 + k * outStride2;
                    size_t batch1Idx =
                        batch * batch1Stride0 + m * batch1Stride1;
                    size_t batch2Idx =
                        batch * batch2Stride0 + k * batch2Stride2;

                    outVecs[0] = Avx2Traits<T>::empty();

                    for (size_t n = 0; n < N; n++) {
                        Vec batch1Vec =
                            Avx2Traits<T>::set1(batch1Data[batch1Idx]);

                        batch2Vecs[0] = Avx2Traits<T>::load(
                            batch2Data + batch2Idx);
                        outVecs[0] = Avx2Traits<T>::fma(
                            batch1Vec, batch2Vecs[0], outVecs[0]);

                        batch1Idx += batch1Stride2;
                        batch2Idx += batch2Stride1;
                    }

                    Avx2Traits<T>::store(outData + outIdx, outVecs[0]);
                }

                for (; k < K; k++) {
                    size_t outIdx =
                        batch * outStride0 + m * outStride1 + k * outStride2;
                    size_t batch1Idx =
                        batch * batch1Stride0 + m * batch1Stride1;
                    size_t batch2Idx =
                        batch * batch2Stride0 + k * batch2Stride2;
                    outData[outIdx] = 0;

                    for (size_t n = 0; n < N; n++) {
                        outData[outIdx] +=
                            batch1Data[batch1Idx] * batch2Data[batch2Idx];

                        batch1Idx += batch1Stride2;
                        batch2Idx += batch2Stride1;
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
