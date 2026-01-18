#pragma once
#include "tityos/ty/cuda/cuda_import.h"
#include "tityos/ty/tensor/BaseTensor.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ty {
namespace internal {

    template <typename T> struct CudaTensorView {
        T* data;
        size_t* shape;
        ptrdiff_t* strides;
        size_t offset;
        int ndim;

        __host__ __device__ T& operator[](size_t linear_idx) const {
            size_t memIdx = offset;
            size_t idx = linear_idx;

            for (int d = ndim - 1; d >= 0; d--) {
                size_t i = idx % shape[d];
                idx /= shape[d];
                memIdx += i * strides[d];
            }

            return data[memIdx];
        }
    };

    template <typename T>
    CudaTensorView<T> buildCudaTensorView(const BaseTensor& tensor) {
        CudaTensorView<T> tensorView;

        tensorView.data =
            reinterpret_cast<T*>(tensor.getTensorStorage()->begin());

        size_t* d_shape;
        ptrdiff_t* d_strides;

        size_t ndim = tensor.getNDim();

        // copy shape and strides to gpu
        cudaMalloc(&d_shape, sizeof(T) * ndim);
        cudaMalloc(&d_strides, sizeof(T) * ndim);

        cudaMemcpy(d_shape, tensor.getShape().data(), sizeof(size_t) * ndim,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, tensor.getStrides().data(),
                   sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice);

        tensorView.shape = d_shape;
        tensorView.strides = d_strides;
        tensorView.offset = tensor.getLayout().getOffset();
        tensorView.ndim = ndim;

        return tensorView;
    }

} // namespace internal
} // namespace ty
