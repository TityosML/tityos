#include "tityos/ty/ops/cuda/bmm.h"

namespace ty {
namespace internal {
    BaseTensor backend::CUDABackend::bmm(const BaseTensor& batch1,
                                         const BaseTensor& batch2) {

        if (batch1.getDType() != batch2.getDType()) {
            throw std::invalid_argument("Types must match for batch matrix-matrix product");
        }

        // TODO: Add shape checking for inputs (b, n, m), (b, m, p)

        // TODO: Replace this with the copy function when that gets updated
        auto resultStorage = std::make_shared<TensorStorage>(
            batch1.getLogicalSize(), batch1.getDevice());
        ShapeStrides resultLayout(batch1.getShape(), batch1.getNDim());
        BaseTensor result(resultStorage, resultLayout, batch1.getDType());

        switch (batch1.getDType()) {
        case DType::Int8:
            launchBMMKernel<int8_t>(batch1, batch2, result);
            break;
        case DType::UInt8:
            launchBMMKernel<uint8_t>(batch1, batch2, result);
            break;
        case DType::Int16:
            launchBMMKernel<int16_t>(batch1, batch2, result);
            break;
        case DType::UInt16:
            launchBMMKernel<uint16_t>(batch1, batch2, result);
            break;
        case DType::Int32:
            launchBMMKernel<int32_t>(batch1, batch2, result);
            break;
        case DType::UInt32:
            launchBMMKernel<uint32_t>(batch1, batch2, result);
            break;
        case DType::Int64:
            launchBMMKernel<int64_t>(batch1, batch2, result);
            break;
        case DType::UInt64:
            launchBMMKernel<uint64_t>(batch1, batch2, result);
            break;
        case DType::Float32:
            launchBMMKernel<float>(batch1, batch2, result);
            break;
        case DType::Float64:
            launchBMMKernel<double>(batch1, batch2, result);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for batch matrix-matrix product");
        }

        return result;
    }
} // namespace internal
} // namespace ty
