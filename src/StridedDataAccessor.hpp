#pragma once
#include <ByteArray.hpp>
#include <Dtype.hpp>
#include <ShapeStrides.hpp>

#include <memory>

class StridedDataAccessor {
private:
    std::shared_ptr<ByteArray> byteArray;
    ShapeStrides layout;

public:
    StridedDataAccessor(
        std::shared_ptr<ByteArray> data,
        const ShapeStrides& layout_
    ) : byteArray(std::move(data)), layout(layout_) {}

    void* at(const std::array<size_t, MAX_DIMS>& index) const {
        size_t byteOffset = layout.computeByteIndex(index);
        return byteArray->at(byteOffset);
    }

    const ShapeStrides& getLayout() const { return layout; }
    const std::shared_ptr<ByteArray>& getByteArray() const { return byteArray; }
};