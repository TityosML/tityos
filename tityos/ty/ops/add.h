#pragma once

#include "tityos/ty/backend/Backend.h"
#include "tityos/ty/export.h"
#include "tityos/ty/ops/expand.h"
#include "tityos/ty/ops/utils.h"

#include <memory>

namespace ty {
Tensor TITYOS_API add(const Tensor& tensor1, const Tensor& tensor2);
}