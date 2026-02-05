#pragma once

#include "tityos/ty/tensor/Slice.h"

#include <initializer_list>
#include <variant>

namespace ty {
using IndexItem = std::variant<Slice, ptrdiff_t>;
using IndexList = std::initializer_list<IndexItem>;
} // namespace ty