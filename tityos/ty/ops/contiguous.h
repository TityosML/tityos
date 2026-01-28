#pragma once

#include "tityos/ty/backend/Backend.h"
#include "tityos/ty/ops/expand.h"
#include "tityos/ty/ops/utils.h"

#include <memory>

namespace ty {
Tensor contiguous(const Tensor& tensor);
}