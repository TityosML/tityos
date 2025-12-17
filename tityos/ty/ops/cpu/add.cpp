#include "tityos/ty/ops/cpu/add.h"

#include <cstdint>
#include <stdexcept>

namespace ty {
namespace internal {
    template <typename T>
    void addCpuKernel(Tensor& result, const Tensor& tensor1,
                      const Tensor& tensor2) {
        auto resultIt = result.begin();
        auto it1 = tensor1.begin();
        auto it2 = tensor2.begin();

        while (resultIt != result.end()) {
            *(reinterpret_cast<T*>(*resultIt)) =
                *(reinterpret_cast<const T*>(*it1)) +
                *(reinterpret_cast<const T*>(*it2));
            ++resultIt;
            ++it1;
            ++it2;
        }
    }

    void internalAddCpu(Tensor& result, const Tensor& tensor1,
                        const Tensor& tensor2) {
        if (result.getDType() != tensor1.getDType() ||
            result.getDType() != tensor2.getDType()) {
            throw std::invalid_argument(
                "Types must match for addition");
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
    }
} // namespace internal
} // namespace ty
