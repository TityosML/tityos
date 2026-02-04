#include "tityos/ty/ops/cpu/contiguous.h"

namespace ty {
namespace internal {
    template <typename T>
    void contiguousCpuKernel(const TensorView<T>& outView,
                             const TensorView<T>& tensorView) {
        size_t total =
            std::accumulate(outView.shape, outView.shape + outView.ndim, 1,
                            std::multiplies<size_t>());

        T* __restrict__ outData = outView.data + outView.offset;
        const T* __restrict__ tensorData = tensorView.data + tensorView.offset;

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

        switch (result.getDType()) {
        case DType::Int8:
            contiguousCpuKernel<int8_t>(result, tensor);
            break;
        case DType::UInt8:
            contiguousCpuKernel<uint8_t>(result, tensor);
            break;
        case DType::Int16:
            contiguousCpuKernel<int16_t>(result, tensor);
            break;
        case DType::UInt16:
            contiguousCpuKernel<uint16_t>(result, tensor);
            break;
        case DType::Int32:
            contiguousCpuKernel<int32_t>(result, tensor);
            break;
        case DType::UInt32:
            contiguousCpuKernel<uint32_t>(result, tensor);
            break;
        case DType::Int64:
            contiguousCpuKernel<int64_t>(result, tensor);
            break;
        case DType::UInt64:
            contiguousCpuKernel<uint64_t>(result, tensor);
            break;
        case DType::Float32:
            contiguousCpuKernel<float>(result, tensor);
            break;
        case DType::Float64:
            contiguousCpuKernel<double>(result, tensor);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for addition");
        }

        return result;
    }
}; // namespace internal
} // namespace ty