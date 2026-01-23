#pragma once

#include "tityos/ty/tensor/BaseTensor.h"

namespace ty {
namespace internal {
    template <typename T> struct TensorView {
        T* data;
        const size_t* shape;
        const ptrdiff_t* strides;
        size_t ndim;
        size_t offset;

        TensorView(const BaseTensor& baseTensor)
            : data(baseTensor.elemAt<T>((size_t) 0)), shape(baseTensor.getShape().data()),
              strides(baseTensor.getStrides().data()), ndim(baseTensor.getNDim()),
              offset(baseTensor.getLayout().getOffset()) {}
    };
} // namespace internal
} // namespace ty