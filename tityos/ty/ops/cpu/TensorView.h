#pragma once

#include "tityos/ty/tensor/BaseTensor.h"

namespace ty {
namespace internal {
    struct TensorView {
        void* data;
        const size_t* shape;
        const ptrdiff_t* strides;
        size_t ndim;
        size_t offset;

        TensorView(const BaseTensor& baseTensor)
            : data(baseTensor.at((size_t)0)), shape(baseTensor.getShape().data()),
              strides(baseTensor.getStrides().data()), ndim(baseTensor.getNDim()),
              offset(baseTensor.getLayout().getOffset()) {}
    };
} // namespace internal
} // namespace ty