#include "tityos/ty/ops/cuda/contiguous.h"

namespace ty {
namespace internal {
    BaseTensor backend::CUDABackend::contiguous(const BaseTensor& tensor) {
        auto result = emptyLike(tensor);

        switch (tensor.getDType()) {
        case DType::Int8:
            launchContiguousKernel<int8_t>(tensor, result);
            break;
        case DType::UInt8:
            launchContiguousKernel<uint8_t>(tensor, result);
            break;
        case DType::Int16:
            launchContiguousKernel<int16_t>(tensor, result);
            break;
        case DType::UInt16:
            launchContiguousKernel<uint16_t>(tensor, result);
            break;
        case DType::Int32:
            launchContiguousKernel<int32_t>(tensor, result);
            break;
        case DType::UInt32:
            launchContiguousKernel<uint32_t>(tensor, result);
            break;
        case DType::Int64:
            launchContiguousKernel<int64_t>(tensor, result);
            break;
        case DType::UInt64:
            launchContiguousKernel<uint64_t>(tensor, result);
            break;
        case DType::Float32:
            launchContiguousKernel<float>(tensor, result);
            break;
        case DType::Float64:
            launchContiguousKernel<double>(tensor, result);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for contiguous");
        }

        return result;
    }
} // namespace internal
} // namespace ty
