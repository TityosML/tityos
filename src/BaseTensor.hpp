#pragma once

#include <ShapeStrides.hpp>
#include <StridedDataAccessor.hpp>
#include <Dtype.hpp>

class BaseTensor {
    private:
        DType dtype;
        StridedDataAccessor dataAccessor;
};