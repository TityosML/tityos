#include "tityos/ty/ops/cpu/add.h"

#include <iostream>

namespace ty {
namespace internal {
    template <typename T>
    void addCpuKernel(const TensorView<T>& outView,
                      const TensorView<T>& tensor1View,
                      const TensorView<T>& tensor2View) {
        size_t total =
            std::accumulate(outView.shape, outView.shape + outView.ndim, 1,
                            std::multiplies<size_t>());

        T* __restrict__ outData = outView.data + outView.offset;
        const T* __restrict__ tensor1Data =
            tensor1View.data + tensor1View.offset;
        const T* __restrict__ tensor2Data =
            tensor2View.data + tensor2View.offset;

        size_t idx[MAX_DIMS];
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

    BaseTensor backend::CPUBackend::add(const BaseTensor& tensor1,
                                        const BaseTensor& tensor2) {
        // TODO: Replace this with the copy function when that gets updated
        auto resultStorage = std::make_shared<TensorStorage>(
            tensor1.getLogicalSize(), tensor1.getDevice());
        ShapeStrides resultLayout(tensor1.getShape(), tensor1.getNDim());
        BaseTensor result(resultStorage, resultLayout, tensor1.getDType());

        // Avx2 Optimized kernel
        if (tensor1.isContiguous() && tensor2.isContiguous()) {
            addAvx2(result, tensor1, tensor2);
            return result;
        }

        switch (result.getDType()) {
        case DType::Int8:
            addCpuKernel<int8_t>(result, tensor1, tensor2);
            break;
        case DType::UInt8:
            addCpuKernel<uint8_t>(result, tensor1, tensor2);
            break;
        case DType::Int16:
            addCpuKernel<int16_t>(result, tensor1, tensor2);
            break;
        case DType::UInt16:
            addCpuKernel<uint16_t>(result, tensor1, tensor2);
            break;
        case DType::Int32:
            addCpuKernel<int32_t>(result, tensor1, tensor2);
            break;
        case DType::UInt32:
            addCpuKernel<uint32_t>(result, tensor1, tensor2);
            break;
        case DType::Int64:
            addCpuKernel<int64_t>(result, tensor1, tensor2);
            break;
        case DType::UInt64:
            addCpuKernel<uint64_t>(result, tensor1, tensor2);
            break;
        case DType::Float32:
            addCpuKernel<float>(result, tensor1, tensor2);
            break;
        case DType::Float64:
            addCpuKernel<double>(result, tensor1, tensor2);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for addition");
        }

        return result;
    }
} // namespace internal
} // namespace ty
