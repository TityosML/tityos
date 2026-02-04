#include "tityos/ty/ops/cuda/add.h"

namespace ty {
namespace internal {
    BaseTensor backend::CUDABackend::add(const BaseTensor& tensor1,
                                         const BaseTensor& tensor2) {

        if (tensor1.getDType() != tensor2.getDType()) {
            throw std::invalid_argument("Types must match for addition");
        }

        auto result = emptyLike(tensor1);

        switch (tensor1.getDType()) {
        case DType::Int8:
            launchAddKernel<int8_t>(tensor1, tensor2, result);
            break;
        case DType::UInt8:
            launchAddKernel<uint8_t>(tensor1, tensor2, result);
            break;
        case DType::Int16:
            launchAddKernel<int16_t>(tensor1, tensor2, result);
            break;
        case DType::UInt16:
            launchAddKernel<uint16_t>(tensor1, tensor2, result);
            break;
        case DType::Int32:
            launchAddKernel<int32_t>(tensor1, tensor2, result);
            break;
        case DType::UInt32:
            launchAddKernel<uint32_t>(tensor1, tensor2, result);
            break;
        case DType::Int64:
            launchAddKernel<int64_t>(tensor1, tensor2, result);
            break;
        case DType::UInt64:
            launchAddKernel<uint64_t>(tensor1, tensor2, result);
            break;
        case DType::Float32:
            launchAddKernel<float>(tensor1, tensor2, result);
            break;
        case DType::Float64:
            launchAddKernel<double>(tensor1, tensor2, result);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for addition");
        }

        return result;
    }
} // namespace internal
} // namespace ty
