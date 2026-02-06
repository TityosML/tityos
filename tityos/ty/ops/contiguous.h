#pragma once

#include "tityos/ty/backend/Backend.h"
#include "tityos/ty/ops/expand.h"
#include "tityos/ty/ops/utils.h"

#include <memory>

namespace ty {
namespace internal {
    BaseTensor contiguous(const BaseTensor& tensor);
}

Tensor contiguous(const Tensor& tensor);
} // namespace ty