#pragma once

#include "tityos/ty/tensor/Slice.h"

#include <initializer_list>
#include <variant>
#include <vector>

namespace ty {
namespace internal {
    class BaseTensor;
}
struct TITYOS_API BoolMask {
    const internal::BaseTensor* boolTensor;
};
using IndexItem = std::variant<Slice, ptrdiff_t, BoolMask>;
using IndexList = std::vector<IndexItem>;
} // namespace ty