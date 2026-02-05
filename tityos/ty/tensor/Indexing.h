#pragma once
#include "tityos/ty/tensor/Index.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <vector>

namespace ty {
namespace internal {
    struct IndexingResult {
        ty::internal::ShapeStrides newLayout;
        // TODO: mask info
    };

    IndexingResult resolveIndices(const ty::IndexList& indices,
                                  const ShapeStrides& layout);

    ShapeStrides applySlice(const ShapeStrides& layout, size_t dim,
                            const Slice& slice);
    ShapeStrides applySelect(const ShapeStrides& layout, size_t dim,
                             ptrdiff_t select);
    // TODO: applyMask
} // namespace internal
} // namespace ty