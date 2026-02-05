#pragma once
#include "tityos/ty/tensor/Index.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <vector>

namespace ty {
namespace internal {
    struct IndexingResult {
        ty::internal::ShapeStrides newLayout;
        std::vector<size_t> gather;
    };

    struct MaskInfo {
        const bool* data;
        ShapeStrides layout;
    };

    IndexingResult resolveIndices(const ty::IndexList& indices,
                                  const ShapeStrides& layout,
                                  std::optional<MaskInfo> mask);

    IndexingResult applyMask(const ShapeStrides& layout, const MaskInfo& mask);

    ShapeStrides applySlice(const ShapeStrides& layout, size_t dim,
                            const Slice& slice);
    ShapeStrides applySelect(const ShapeStrides& layout, size_t dim,
                             ptrdiff_t select);
} // namespace internal
} // namespace ty