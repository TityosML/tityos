#pragma once

#include "tityos/ty/cuda/cuda_import.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace ty {
namespace cuda {
    inline bool isCudaAvailable() {
#ifdef TITYOS_USE_CUDA
        // TODO: Change this to check for device availablilty
        return true;
#else
        return false;
#endif
    }

} // namespace cuda
} // namespace ty
