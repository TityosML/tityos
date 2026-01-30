#include "tityos/ty/ops/cpu/bmm.h"

namespace ty {
namespace internal {
    template <typename T>
    void bmmCpuKernel(const TensorView<T>& outView,
                      const TensorView<T>& tensor1View,
                      const TensorView<T>& tensor2View) {
        T* __restrict__ outData = outView.data + outView.offset;
        const T* __restrict__ tensor1Data =
            tensor1View.data + tensor1View.offset;
        const T* __restrict__ tensor2Data =
            tensor2View.data + tensor2View.offset;

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

        ShapeStrides resultLayout(
            ty::internal::TensorShape{shape1[0], shape1[1], shape2[2]}, 3);
        auto resultStorage = std::make_shared<TensorStorage>(
            resultLayout.numElements() * dtypeSize(batch1.getDType()),
            batch1.getDevice());
        BaseTensor result(resultStorage, resultLayout, batch1.getDType());

        // Avx2 Optimized kernel
        if (batch1.isContiguous() && batch2.isContiguous()) {
            bmmAvx2(result, batch1, batch2);
            return result;
        }

        switch (result.getDType()) {
        case DType::Int8:
            bmmCpuKernel<int8_t>(result, batch1, batch2);
            break;
        case DType::UInt8:
            bmmCpuKernel<uint8_t>(result, batch1, batch2);
            break;
        case DType::Int16:
            bmmCpuKernel<int16_t>(result, batch1, batch2);
            break;
        case DType::UInt16:
            bmmCpuKernel<uint16_t>(result, batch1, batch2);
            break;
        case DType::Int32:
            bmmCpuKernel<int32_t>(result, batch1, batch2);
            break;
        case DType::UInt32:
            bmmCpuKernel<uint32_t>(result, batch1, batch2);
            break;
        case DType::Int64:
            bmmCpuKernel<int64_t>(result, batch1, batch2);
            break;
        case DType::UInt64:
            bmmCpuKernel<uint64_t>(result, batch1, batch2);
            break;
        case DType::Float32:
            bmmCpuKernel<float>(result, batch1, batch2);
            break;
        case DType::Float64:
            bmmCpuKernel<double>(result, batch1, batch2);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for addition");
        }

        return result;
    }
}; // namespace internal
} // namespace ty