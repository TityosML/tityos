#pragma once
#include "tityos/ty/tensor/BaseTensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace ty {
namespace internal {

    template <typename T> struct CudaTensorView {
        T* data;
        size_t shape[MAX_DIMS];
        ptrdiff_t strides[MAX_DIMS];
        size_t offset;
        size_t ndim;

        __device__ T& operator[](size_t mem_offset) const {
            return data[offset + mem_offset];
        }

        __device__ __forceinline__ T& atLinear(size_t linear_idx) const {
            ptrdiff_t memIdx = offset;
            size_t idx = linear_idx;

            for (int d = ndim - 1; d >= 0; d--) {
                size_t i = idx % shape[d];
                idx /= shape[d];
                memIdx += i * strides[d];
            }

            return data[memIdx];
        }

        // Prereq: ndim = 3, pass variables to reduce repeated array accesses
        __device__ __forceinline__ T& at3d(int x, int y, int z, int strideX,
                                           int strideY, int strideZ) const {
            return data[offset + (x * strideX + y * strideY + z * strideZ)];
        }
    };

    template <typename T>
    CudaTensorView<T> buildCudaTensorView(const BaseTensor& tensor) {
        CudaTensorView<T> tensorView;

        tensorView.data =
            reinterpret_cast<T*>(tensor.getTensorStorage()->begin());

        size_t ndim = tensor.getNDim();
        TensorShape shape = tensor.getShape();
        TensorStrides strides = tensor.getStrides();

        for (size_t i = 0; i < ndim; i++) {
            tensorView.shape[i] = shape[i];
            tensorView.strides[i] = strides[i];
        }

        tensorView.offset = tensor.getLayout().getOffset();
        tensorView.ndim = ndim;

        return tensorView;
    }

} // namespace internal
} // namespace ty
