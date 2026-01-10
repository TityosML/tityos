#include "tityos/ty/ops/cpu/add.h"

#include <iostream>

namespace ty {
namespace internal {
    template <typename T>
    void addCpuKernel(T* __restrict__ out, const T* __restrict__ tensor1,
                      const T* __restrict__ tensor2, const size_t* shape,
                      const ptrdiff_t* outStride,
                      const ptrdiff_t* tensor1Stride,
                      const ptrdiff_t* tensor2Stride, size_t ndim,
                      ptrdiff_t outOffset = 0, ptrdiff_t tensor1Offset = 0,
                      ptrdiff_t tensor2Offset = 0) {
        size_t total =
            std::accumulate(shape, shape + ndim, 1, std::multiplies<size_t>());

        T* __restrict__ outBase = out + outOffset;
        const T* __restrict__ tensor1Base = tensor1 + tensor1Offset;
        const T* __restrict__ tensor2Base = tensor2 + tensor2Offset;

        size_t idx[MAX_DIMS];
        for (size_t i = 0; i < ndim; i++) {
            idx[i] = 0;
        }

        for (size_t linear = 0; linear < total; linear++) {
            ptrdiff_t outIdx = 0;
            ptrdiff_t tensor1Idx = 0;
            ptrdiff_t tensor2Idx = 0;

            for (size_t i = 0; i < ndim; i++) {
                outIdx += idx[i] * outStride[i];
                tensor1Idx += idx[i] * tensor1Stride[i];
                tensor2Idx += idx[i] * tensor2Stride[i];
            }

            outBase[outIdx] = tensor1Base[tensor1Idx] + tensor2Base[tensor2Idx];

            for (size_t i = ndim; i-- > 0;) {
                if (++idx[i] < shape[i]) {
                    break;
                }

                idx[i] = 0;
            }
        }
    }

    BaseTensor internalAddCpu(const BaseTensor& tensor1,
                              const BaseTensor& tensor2) {
        // TODO: Replace this with the copy function when that gets updated
        auto resultStorage = std::make_shared<TensorStorage>(
            tensor1.getLogicalSize(), tensor1.getDevice());
        ShapeStrides resultLayout(tensor1.getShape(), tensor1.getNDim());
        BaseTensor result(resultStorage, resultLayout, tensor1.getDType());

        if (result.getDType() != tensor1.getDType() ||
            result.getDType() != tensor2.getDType()) {
            throw std::invalid_argument("Types must match for addition");
        }

        switch (result.getDType()) {
        case DType::Int8:
            addCpuKernel<int8_t>(
                result.elemAt<int8_t>(size_t(0)),
                tensor1.elemAt<int8_t>(size_t(0)),
                tensor2.elemAt<int8_t>(size_t(0)), result.getShape().data(),
                result.getStrides().data(), tensor1.getStrides().data(),
                tensor2.getStrides().data(), result.getNDim());
            break;
        case DType::UInt8:
            addCpuKernel<uint8_t>(
                result.elemAt<uint8_t>(size_t(0)),
                tensor1.elemAt<uint8_t>(size_t(0)),
                tensor2.elemAt<uint8_t>(size_t(0)), result.getShape().data(),
                result.getStrides().data(), tensor1.getStrides().data(),
                tensor2.getStrides().data(), result.getNDim());
            break;
        case DType::Int16:
            addCpuKernel<int16_t>(
                result.elemAt<int16_t>(size_t(0)),
                tensor1.elemAt<int16_t>(size_t(0)),
                tensor2.elemAt<int16_t>(size_t(0)), result.getShape().data(),
                result.getStrides().data(), tensor1.getStrides().data(),
                tensor2.getStrides().data(), result.getNDim());
            break;
        case DType::UInt16:
            addCpuKernel<uint16_t>(
                result.elemAt<uint16_t>(size_t(0)),
                tensor1.elemAt<uint16_t>(size_t(0)),
                tensor2.elemAt<uint16_t>(size_t(0)), result.getShape().data(),
                result.getStrides().data(), tensor1.getStrides().data(),
                tensor2.getStrides().data(), result.getNDim());
            break;
        case DType::Int32:
            addCpuKernel<int32_t>(
                result.elemAt<int32_t>(size_t(0)),
                tensor1.elemAt<int32_t>(size_t(0)),
                tensor2.elemAt<int32_t>(size_t(0)), result.getShape().data(),
                result.getStrides().data(), tensor1.getStrides().data(),
                tensor2.getStrides().data(), result.getNDim());
            break;
        case DType::UInt32:
            addCpuKernel<uint32_t>(
                result.elemAt<uint32_t>(size_t(0)),
                tensor1.elemAt<uint32_t>(size_t(0)),
                tensor2.elemAt<uint32_t>(size_t(0)), result.getShape().data(),
                result.getStrides().data(), tensor1.getStrides().data(),
                tensor2.getStrides().data(), result.getNDim());
            break;
        case DType::Int64:
            addCpuKernel<int64_t>(
                result.elemAt<int64_t>(size_t(0)),
                tensor1.elemAt<int64_t>(size_t(0)),
                tensor2.elemAt<int64_t>(size_t(0)), result.getShape().data(),
                result.getStrides().data(), tensor1.getStrides().data(),
                tensor2.getStrides().data(), result.getNDim());
            break;
        case DType::UInt64:
            addCpuKernel<uint64_t>(
                result.elemAt<uint64_t>(size_t(0)),
                tensor1.elemAt<uint64_t>(size_t(0)),
                tensor2.elemAt<uint64_t>(size_t(0)), result.getShape().data(),
                result.getStrides().data(), tensor1.getStrides().data(),
                tensor2.getStrides().data(), result.getNDim());
            break;
        case DType::Float32:
            addCpuKernel<float>(
                result.elemAt<float>(size_t(0)),
                tensor1.elemAt<float>(size_t(0)),
                tensor2.elemAt<float>(size_t(0)), result.getShape().data(),
                result.getStrides().data(), tensor1.getStrides().data(),
                tensor2.getStrides().data(), result.getNDim());
            break;
        case DType::Float64:
            addCpuKernel<double>(
                result.elemAt<double>(size_t(0)),
                tensor1.elemAt<double>(size_t(0)),
                tensor2.elemAt<double>(size_t(0)), result.getShape().data(),
                result.getStrides().data(), tensor1.getStrides().data(),
                tensor2.getStrides().data(), result.getNDim());
            break;
        default:
            throw std::runtime_error("Unsupported dtype for addition");
        }

        return result;
    }
} // namespace internal
} // namespace ty
