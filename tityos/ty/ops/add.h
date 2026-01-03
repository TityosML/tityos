#pragma once

#include "tityos/ty/ops/core/add.h"
#include "tityos/ty/ops/utils.h"
#include "tityos/ty/ops/expand.h"

#include <memory>

namespace ty {
Tensor add(const Tensor& tensor1, const Tensor& tensor2);
}