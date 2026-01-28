#pragma once

#include "tityos/ty/Device.h"
#include "tityos/ty/export.h"

#include <cstddef>
#include <mutex>
#include <stdexcept>

#ifdef _WIN32
    #include <windows.h>
    #define LIB_EXTENSION ".dll"
#else
    #include <dlfcn.h>
    #ifdef __APPLE__
        #define LIB_EXTENSION ".dylib"
    #else
        #define LIB_EXTENSION ".so"
    #endif
#endif

namespace ty {
namespace internal {
    class BaseTensor;

    namespace backend {
        class Backend;

        extern Backend* cpuBackend;
        extern Backend* cudaBackend;
        extern std::mutex mtx;

        class Backend {
          public:
            Backend() = default;
            virtual ~Backend() = default;

            virtual void* allocate(size_t bytes, int index) = 0;
            virtual void deallocate(void* ptr) = 0;
            virtual void copyData(void* destPtr, const void* srcPtr,
                                  size_t numBytes) = 0;
            virtual void copyDataFromCpu(void* destPtr, const void* srcPtr,
                                         size_t numBytes) = 0;

            virtual BaseTensor add(const BaseTensor& tensor1,
                                   const BaseTensor& tensor2) = 0;
            virtual BaseTensor bmm(const BaseTensor& batch1,
                                   const BaseTensor& batch2) = 0;
            virtual BaseTensor contiguous(const BaseTensor& tensor) = 0;
            virtual BaseTensor toCpu(const BaseTensor& tensor) = 0;
        };

        Backend* getBackend(DeviceType type);

        bool tryLoadCpuBackend();
        bool tryLoadCudaBackend();
    } // namespace backend
} // namespace internal
} // namespace ty
