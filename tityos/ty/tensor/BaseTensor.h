#pragma once

#include "tityos/ty/tensor/ShapeStrides.h"
#include "tityos/ty/tensor/StridedDataAccessor.h"
#include "tityos/ty/tensor/Dtype.h"

class BaseTensor {
    private:
        DType dtype;
        StridedDataAccessor dataAccessor;
};