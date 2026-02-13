#pragma once

#include "tityos/ty/backend/Backend.h"
#include "tityos/ty/export.h"
#include "tityos/ty/ops/empty.h"
#include "tityos/ty/ops/expand.h"
#include "tityos/ty/ops/reshape.h"
#include "tityos/ty/ops/utils.h"
#include "tityos/ty/tensor/Tensor.h"

#include <memory>

namespace ty {
namespace internal {
    BaseTensor matmul(const BaseTensor& tensor1, const BaseTensor& tensor2);

    BaseTensor bmm(const BaseTensor& tensor1, const BaseTensor& tensor2);
} // namespace internal

Tensor TITYOS_API matmul(const Tensor& tensor1, const Tensor& tensor2);

Tensor TITYOS_API bmm(const Tensor& tensor1, const Tensor& tensor2);
} // namespace ty