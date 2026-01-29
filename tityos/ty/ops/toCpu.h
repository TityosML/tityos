#pragma once

#include "tityos/ty/backend/Backend.h"
#include "tityos/ty/ops/expand.h"
#include "tityos/ty/ops/utils.h"

#include <memory>

namespace ty {
Tensor toCpu(const Tensor& tensor);
}