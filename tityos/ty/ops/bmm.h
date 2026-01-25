#pragma once

#include "tityos/ty/ops/utils.h"
#include "tityos/ty/tensor/Tensor.h"
#include "tityos/ty/backend/Backend.h"

#include <memory>

namespace ty {
Tensor bmm(const Tensor& tensor1, const Tensor& tensor2);
}