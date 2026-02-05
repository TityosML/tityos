#include "tityos/ty/tensor/Indexing.h"

#include <stdexcept>
#include <vector>

namespace ty {
namespace internal {
    IndexingResult resolveIndices(const IndexList& indices,
                                  const ShapeStrides& layout,
                                  std::optional<MaskInfo> mask) {
        if (mask.has_value()) {
            return applyMask(layout, mask.value());
        }

        ShapeStrides newLayout = layout;

        size_t dim = 0;

        for (const auto& idx : indices) {
            if (dim >= newLayout.getNDim()) {
                throw std::out_of_range("Invalid index");
            }

            if (std::holds_alternative<Slice>(idx)) {
                const Slice& slice = std::get<Slice>(idx);
                newLayout =
                    newLayout.slice(dim, slice.start, slice.stop, slice.step);
                dim++;
            } else {
                ptrdiff_t select = std::get<ptrdiff_t>(idx);
                newLayout = newLayout.select(dim, select);
            }
        }

        return IndexingResult{std::move(newLayout), {}};
    }

    IndexingResult applyMask(const ShapeStrides& layout, const MaskInfo& mask) {
        if (mask.layout.getShape() != layout.getShape()) {
            throw std::invalid_argument("Invalid mask.");
        };

        size_t nonZeros = 0;
        size_t total = layout.numElements();
        for (size_t i = 0; i < total; i++) {
            if (mask.data[i]) {
                nonZeros++;
            }
        }

        std::vector<size_t> gatherIndices;
        gatherIndices.reserve(nonZeros);
        for (size_t i = 0; i < total; i++) {
            if (mask.data[i]) {
                size_t offset = static_cast<size_t>(layout.linearToOffset(i));
                gatherIndices.push_back(offset);
            }
        }

        return IndexingResult{ShapeStrides({nonZeros}, 1),
                              std::move(gatherIndices)};
    }
    
    ShapeStrides applySlice(const ShapeStrides& layout, size_t dim,
                            const Slice& slice) {
        return layout.slice(dim, slice.start, slice.stop, slice.step);
    }

    ShapeStrides applySelect(const ShapeStrides& layout, size_t dim,
                             ptrdiff_t select) {
        return layout.select(dim, select);
    }

} // namespace internal
} // namespace ty