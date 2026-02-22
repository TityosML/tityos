#include "tityos/ty/tensor/Indexing.h"

#include "tityos/ty/tensor/BaseTensor.h"

#include <stdexcept>
#include <vector>

namespace ty {
namespace internal {
    IndexingResult resolveIndices(const IndexList& indices, const BaseTensor& data) {
        ShapeStrides newLayout = data.getLayout();

        size_t dim = 0;

        for (const auto& idx : indices) {
            if (dim >= newLayout.getNDim()) {
                throw std::out_of_range("Invalid index");
            }

            if (std::holds_alternative<Slice>(idx)) {
                const Slice& slice = std::get<Slice>(idx);
                newLayout = newLayout.slice(dim, slice.start, slice.stop, slice.step);
                dim++;
            }
            if (std::holds_alternative<ptrdiff_t>(idx)) {
                ptrdiff_t select = std::get<ptrdiff_t>(idx);
                newLayout = newLayout.select(dim, select);
            }
            if (std::holds_alternative<BoolMask>(idx)) {
                return applyMask(data, std::get<BoolMask>(idx));
            }
        }

        return IndexingResult{std::move(newLayout), {}};
    }

    IndexingResult applyMask(const BaseTensor& data, const BoolMask& mask) {
        const ShapeStrides layout = data.getLayout();
        const BaseTensor& maskTensor = *mask.boolTensor;

        if (maskTensor.getShape() != layout.getShape()) {
            throw std::invalid_argument("Invalid mask.");
        };

        size_t total = layout.numElements();

        std::vector<size_t> gather;
        gather.reserve(total);

        for (size_t i = 0; i < total; ++i) {
            const uint8_t* maskVal = static_cast<const uint8_t*>(maskTensor.at(i));
            if (*maskVal != 0) {
                gather.push_back(i);
            }
        }

        return IndexingResult{ShapeStrides{{gather.size()}, {1}, 0, 1}, std::move(gather)};
    }

    BaseTensor copyFromGather(const BaseTensor& data, const IndexingResult& idxResult) {
        std::vector<size_t> gather = *idxResult.gather;
        size_t total = gather.size();
        size_t elemSize = dtypeSize(data.getDType());
        std::vector<uint8_t> buffer(gather.size() * elemSize);

        const ShapeStrides& layout = data.getLayout();
        if (layout.isContiguous()) {
            uint8_t* dataStartPointer =
                reinterpret_cast<uint8_t*>(data.getTensorStorage()->begin()) + (layout.getOffset() * elemSize);

            for (size_t i = 0; i < total; ++i) {
                size_t idx = gather[i];
                std::memcpy(&buffer[i * elemSize], dataStartPointer + idx * elemSize, elemSize);
            }
        } else {
            for (size_t i = 0; i < total; ++i) {
                void* src = data.at(gather[i]);
                std::memcpy(&buffer[i * elemSize], src, elemSize);
            }
        }

        TensorStorage storage(buffer.size(), data.getDevice());
        storage.copyDataFromCpu(buffer.data(), buffer.size());

        return BaseTensor(std::make_shared<TensorStorage>(std::move(storage)), idxResult.newLayout, data.getDType());
    }
} // namespace internal
} // namespace ty