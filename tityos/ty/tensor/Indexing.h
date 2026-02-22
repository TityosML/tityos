#pragma once

#include "tityos/ty/tensor/Index.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <vector>

namespace ty {
namespace internal {
    class BaseTensor;

    struct IndexingResult {
        ty::internal::ShapeStrides newLayout;
        std::optional<std::vector<size_t>> gather;
    };

    IndexingResult resolveIndices(const ty::IndexList& indices, const BaseTensor& data);

    IndexingResult applyMask(const BaseTensor& data, const BoolMask& mask);

    BaseTensor copyFromGather(const BaseTensor& data, const IndexingResult& idxResult);
} // namespace internal
} // namespace ty