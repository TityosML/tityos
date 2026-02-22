#pragma once

#include "tityos/ty/Device.h"
#include "tityos/ty/export.h"

#include <any>
#include <cstddef>

namespace ty {
constexpr size_t TY_NUM_DTYPES = 10;

enum class DType { Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float32, Float64 };

inline size_t dtypeSize(DType dtype) {
    switch (dtype) {
    case DType::Int8:
    case DType::UInt8:
        return 1;
    case DType::Int16:
    case DType::UInt16:
        return 2;
    case DType::Int32:
    case DType::UInt32:
    case DType::Float32:
        return 4;
    case DType::Int64:
    case DType::UInt64:
    case DType::Float64:
        return 8;
    }

    return 0;
}

inline bool isIntegralType(DType dtype) {
    switch (dtype) {
    case DType::Int8:
    case DType::UInt8:
    case DType::Int16:
    case DType::UInt16:
    case DType::Int32:
    case DType::UInt32:
    case DType::Int64:
    case DType::UInt64:
        return true;
    case DType::Float32:
    case DType::Float64:
        return false;
    }

    return false;
}

inline bool isGradientType(DType dtype) {
    switch (dtype) {
    case DType::Int8:
    case DType::UInt8:
    case DType::Int16:
    case DType::UInt16:
    case DType::Int32:
    case DType::UInt32:
    case DType::Int64:
    case DType::UInt64:
        return false;
    case DType::Float32:
    case DType::Float64:
        return true;
    }

    return false;
}

} // namespace ty