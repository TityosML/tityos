#include "tityos/ty/ops/cuda/bmm.h"

namespace ty {
namespace internal {
    BaseTensor backend::CUDABackend::bmm(const BaseTensor& batch1,
                                         const BaseTensor& batch2) {
        auto shape1 = batch1.getShape();
        auto shape2 = batch2.getShape();

        ShapeStrides resultLayout(
            ty::internal::TensorShape{shape1[0], shape1[1], shape2[2]}, 3);
        auto resultStorage = std::make_shared<TensorStorage>(
            resultLayout.numElements() * dtypeSize(batch1.getDType()),
            batch1.getDevice());
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
            throw std::runtime_error(
                "Unsupported dtype for batch matrix-matrix product");
        }

        return result;
    }
} // namespace internal
} // namespace ty
