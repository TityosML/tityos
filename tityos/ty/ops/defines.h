#pragma once

#include <stdexcept>

#define DEFINE_FUNC_DISPATCH(func)                                             \
    template <typename... Args> void func(Device device, Args&&... args) {     \
        if (device.isCuda()) {                                                 \
            if constexpr (func##CudaExists) {                                  \
                func##Cuda(std::forward<Args>(args)...);                       \
            } else {                                                           \
                throw std::runtime_error("Cannot apply function to a cuda "    \
                                         "tensor CUDA is not available");      \
            }                                                                  \
            return;                                                            \
        } else if (device.isCpu()) {                                           \
            if constexpr (func##Avx2Exists) {                                  \
                func##Avx2(std::forward<Args>(args)...);                       \
            } else {                                                           \
                func##Cpu(std::forward<Args>(args)...);                        \
            }                                                                  \
            return;                                                            \
        }                                                                      \
                                                                               \
        throw std::runtime_error("Unknown device");                            \
    }

#define DECLARE_NO_CUDA_DISPATCH_FUNCTION(func)                                \
    constexpr bool func##CudaExists = false;
#define DECLARE_NO_AVX2_DISPATCH_FUNCTION(func)                                \
    constexpr bool func##Avx2Exists = false;

#ifdef TITYOS_USE_CUDA
    #define DECLARE_CUDA_DISPATCH_FUNCTION(func)                               \
        constexpr bool func##CudaExists = true;
#else
    #define DECLARE_CUDA_DISPATCH_FUNCTION(func)                               \
        DECLARE_NO_CUDA_DISPATCH_FUNCTION(func)
#endif

#ifdef TITYOS_USE_AVX2
    #define DECLARE_AVX2_DISPATCH_FUNCTION(func)                               \
        constexpr bool func##Avx2Exists = true;
#else
    #define DECLARE_AVX2_DISPATCH_FUNCTION(func)                               \
        DECLARE_NO_AVX2_DISPATCH_FUNCTION(func)
#endif

#define DECLARE_ALL_DISPATCH_FUNCTION(func)                                    \
    DECLARE_CUDA_DISPATCH_FUNCTION(func)                                       \
    DECLARE_AVX2_DISPATCH_FUNCTION(func)
