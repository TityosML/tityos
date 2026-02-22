#pragma once

#include "tityos/ty/tensor/Dtype.h"

#include <array>
#include <stdexcept>

namespace ty {
namespace internal {
#define DISPATCH_KERNEL_DTYPE_TABLE(tableName, kernel, argtypes)                                                       \
    using kernel##typesig = void(*) argtypes;                                                                          \
    static const auto tableName = std::array<kernel##typesig, ty::TY_NUM_DTYPES>{                                      \
        &kernel<int8_t>,   &kernel<uint8_t>, &kernel<int16_t>,  &kernel<uint16_t>, &kernel<int32_t>,                   \
        &kernel<uint32_t>, &kernel<int64_t>, &kernel<uint64_t>, &kernel<float>,    &kernel<double>};
} // namespace internal
} // namespace ty