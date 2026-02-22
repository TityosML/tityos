#include "tityos/ty/tensor/cuda/TensorStorage.h"

namespace ty {
namespace internal {
    void* backend::CUDABackend::allocate(size_t bytes, int index) {
        CUDA_CHECK(cudaSetDevice(index));
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, bytes));

        if (!ptr) {
            throw std::runtime_error("Unable to allocate memory for CUDA");
        }

        return ptr;
    }

    void backend::CUDABackend::deallocate(void* ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }

    void backend::CUDABackend::copyData(void* destPtr, const void* srcPtr, size_t numBytes) {
        CUDA_CHECK(cudaMemcpy(destPtr, srcPtr, numBytes, cudaMemcpyHostToHost));
    }

    void backend::CUDABackend::copyDataFromCpu(void* destPtr, const void* srcPtr, size_t numBytes) {
        CUDA_CHECK(cudaMemcpy(destPtr, srcPtr, numBytes, cudaMemcpyHostToDevice));
    }
} // namespace internal
} // namespace ty