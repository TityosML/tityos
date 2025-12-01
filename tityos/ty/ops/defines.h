#pragma once

#include <stdexcept>

#define DEFINE_FUNC_DISPATCH(func)                                             \
    template <typename... Args> void func(Device device, Args&&... args) {     \
        if (device.isCpu()) {                                                  \
            FUNC_CPU_DISPATCH(func, std::forward<Args>(args)...);              \
            return;                                                            \
        } else if (device.isCuda()) {                                          \
            FUNC_CUDA_DISPATCH(func, std::forward<Args>(args)...);             \
            return;                                                            \
        }                                                                      \
                                                                               \
        throw std::runtime_error("Unknown device");                            \
    }                                                                          

#ifdef TITYOS_USE_CUDA
    #define FUNC_CUDA_DISPATCH(func, ...) func##Cuda(__VA_ARGS__)
#else
    #define FUNC_CUDA_DISPATCH(func, ...)                                      \
        throw std::runtime_error(                                              \
            "Cannot apply function to a cuda tensor CUDA is not available")
#endif

#define FUNC_CPU_DISPATCH(func, ...) func##Cpu(__VA_ARGS__);

// --- For when we start implementing SIMD ---
// #ifdef TITYOS_USE_AVX2
//     #define FUNC_CPU_DISPATCH(func, ...) func##Avx2(__VA_ARGS__)
// #elif TITYOS_USE_AVX
//     #define FUNC_CPU_DISPATCH(func, ...) func##Avx2(__VA_ARGS__)
// #else
//     #define FUNC_CPU_DISPATCH(func, ...) func##Cpu(__VA_ARGS__);
// #endif
