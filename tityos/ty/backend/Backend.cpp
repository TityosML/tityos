#include "tityos/ty/backend/Backend.h"

namespace ty {
namespace internal {
    namespace backend {
        Backend* cpuBackend = nullptr;
        Backend* cudaBackend = nullptr;
        std::mutex mtx;

        Backend* getBackend(DeviceType type) {
            std::lock_guard<std::mutex> lock(mtx);
            switch (type) {
            case DeviceType::CPU:
                if (!cpuBackend) {
                    if (!tryLoadCpuBackend()) {
                        throw std::runtime_error("Unable to load CPU Backend");
                    }
                }
                return cpuBackend;
            case DeviceType::CUDA:
                if (!cudaBackend) {
                    if (!tryLoadCudaBackend()) {
                        throw std::runtime_error("Unable to load CUDA Backend");
                    }
                }

                return cudaBackend;
            default:
                return nullptr;
            }
        }

        bool tryLoadCpuBackend() {
            const char* libName = "libtityos_cpu" LIB_EXTENSION;
            using Fn = Backend* (*)();

#ifdef _WIN32
            HMODULE handle = LoadLibraryA(libName);
            if (!handle)
                return;
            auto registerFn = reinterpret_cast<Fn>(
                GetProcAddress(handle, "registerCpuBackend"));
#else
            void* handle = dlopen(libName, RTLD_NOW | RTLD_GLOBAL);
            if (!handle) {
                return false;
            }

            auto registerFn =
                reinterpret_cast<Fn>(dlsym(handle, "registerCpuBackend"));
#endif

            if (registerFn) {
                cpuBackend = registerFn();
                return true;
            }

            return false;
        }

        bool tryLoadCudaBackend() {
            const char* libName = "libtityos_cuda" LIB_EXTENSION;
            using Fn = Backend* (*)();

#ifdef _WIN32
            HMODULE handle = LoadLibraryA(libName);
            if (!handle)
                return;
            auto registerFn = reinterpret_cast<Fn>(
                GetProcAddress(handle, "registerCudaBackend"));
#else
            void* handle = dlopen(libName, RTLD_NOW | RTLD_GLOBAL);
            if (!handle) {
                return false;
            }

            auto registerFn =
                reinterpret_cast<Fn>(dlsym(handle, "registerCudaBackend"));
#endif
            if (registerFn) {
                cudaBackend = registerFn();
                return true;
            }

            return false;
        }
    } // namespace backend
} // namespace internal
} // namespace ty