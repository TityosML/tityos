#pragma once

#include "tityos/ty/tensor/Slice.h"

#include <initializer_list>
#include <variant>
#include <vector>

namespace ty {
using IndexItem = std::variant<Slice, ptrdiff_t>;
using IndexList = std::vector<IndexItem>;
} // namespace ty