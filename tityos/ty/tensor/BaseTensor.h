#pragma once

#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"
#include "tityos/ty/tensor/StridedDataAccessor.h"

namespace ty {
    namespace internal {
        class BaseTensor {
          private:
            DType dtype_;
            StridedDataAccessor dataAccessor_;
        };
    } // namespace internal
} // namespace ty