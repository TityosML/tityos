#pragma once

#include "tityos/ty/cuda/cuda_import.h"

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