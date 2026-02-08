#include "tityos/ty/ops/expand.h"

namespace ty {
namespace internal {
    BaseTensor expand(const BaseTensor& tensor,
                      const std::vector<size_t>& newShape) {
        const size_t oldNDim = tensor.getNDim();
        const size_t newNDim = newShape.size();

        const auto& oldShape = tensor.getShape();
        const auto& oldStrides = tensor.getStrides();

        TensorShape newShapeArray;
        TensorStrides newStrides;

        const ptrdiff_t dimOffset =
            static_cast<ptrdiff_t>(oldNDim) - static_cast<ptrdiff_t>(newNDim);

        if (dimOffset > 0) {
            for (ptrdiff_t i = 0; i < dimOffset; i++) {
                if (oldShape[i] != 1) {
                    throw std::invalid_argument("Cannot remove non-singleton "
                                                "dimensions during expansion");
                }
            }
        }

        for (size_t newDim = 0; newDim < newNDim; newDim++) {
            const ptrdiff_t oldDim = static_cast<ptrdiff_t>(newDim) + dimOffset;
            const size_t newSize = newShape[newDim];

            newShapeArray[newDim] = newSize;

            if (oldDim < 0) {
                newStrides[newDim] = 0;
            } else {
                const size_t oldSize = oldShape[oldDim];

                if (oldSize == newSize) {
                    newStrides[newDim] = oldStrides[oldDim];
                } else if (oldSize == 1) {
                    newStrides[newDim] = 0;
                } else {
                    throw std::invalid_argument(
                        "Cannot expand non-singleton dimension to new size");
                }
            }
        }

        const auto offset = tensor.getLayout().getOffset();
        ShapeStrides newLayout(newShapeArray, newStrides, offset, newNDim);

        return BaseTensor(tensor.getTensorStorage(), newLayout,
                          tensor.getDType());
    }
} // namespace internal

Tensor expand(const Tensor& tensor, const std::vector<size_t>& newShape) {
    return Tensor(std::make_shared<internal::BaseTensor>(
        internal::expand(*tensor.getBaseTensor(), newShape)));
}
} // namespace ty