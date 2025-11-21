#include "tityos/ty/tensor/ShapeStrides.h"

namespace ty {
    namespace internal {
        void ShapeStrides::initialStrides(DType dtype) {
            strides_ = std::array<size_t, MAX_DIMS>{};
            int currentStride = 1;

            for (int i = ndim_ - 1; i >= 0; i--) {
                strides_[i] = currentStride * dtypeSize(dtype);
                currentStride *= shape_[i];
            }
        }
    }
}