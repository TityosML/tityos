#include "tityos/ty/tensor/Tensor.h"

#include "tityos/ty/tensor/ShapeStrides.h"

#include <cmath>

namespace ty {
Tensor::Tensor(const Tensor& other) : baseTensor_(other.baseTensor_) {}

Tensor::Tensor(Tensor&& other) noexcept
    : baseTensor_(std::move(other.baseTensor_)) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        baseTensor_ = other.baseTensor_;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        baseTensor_ = std::move(other.baseTensor_);
    }
    return *this;
}

void* Tensor::at(const size_t* indexStart, const size_t indexSize) const {
    if (indexSize != baseTensor_->getLayout().getNDim()) {
        throw std::invalid_argument(
            "Index size mismatch: expected " +
            std::to_string(baseTensor_->getLayout().getNDim()) + ", got " +
            std::to_string(indexSize));
    }

    const std::array<size_t, internal::MAX_DIMS>& shape =
        baseTensor_->getLayout().getShape();
    for (size_t i = 0; i < indexSize; i++) {
        if (indexStart[i] >= shape[i]) {
            throw std::out_of_range(
                "Index out of bounds at dimension " + std::to_string(i) +
                ": index value " + std::to_string(*(indexStart + i)) +
                " is >= shape dimension " + std::to_string(shape[i]));
        }
    }

    return baseTensor_->at(indexStart);
}

void* Tensor::at(const std::vector<size_t>& index) const {
    return at(index.data(), index.size());
}

void* Tensor::at(const std::initializer_list<size_t>& index) const {
    return at(index.begin(), index.size());
}

using Iterator = internal::BaseTensor::Iterator;

Iterator Tensor::begin() {
    return baseTensor_->begin();
}
Iterator Tensor::end() {
    return baseTensor_->end();
}

Iterator Tensor::begin() const {
    return baseTensor_->begin();
}

Iterator Tensor::end() const {
    return baseTensor_->end();
}

std::string Tensor::itemToStringCpu(const void* item, const DType dtype) const {
    switch (dtype) {
    case DType::Int8:
        return std::to_string(*reinterpret_cast<const int8_t*>(item));
    case DType::UInt8:
        return std::to_string(*reinterpret_cast<const uint8_t*>(item));
    case DType::Int16:
        return std::to_string(*reinterpret_cast<const int16_t*>(item));
    case DType::UInt16:
        return std::to_string(*reinterpret_cast<const uint16_t*>(item));
    case DType::Int32:
        return std::to_string(*reinterpret_cast<const int32_t*>(item));
    case DType::UInt32:
        return std::to_string(*reinterpret_cast<const uint32_t*>(item));
    case DType::Int64:
        return std::to_string(*reinterpret_cast<const int64_t*>(item));
    case DType::UInt64:
        return std::to_string(*reinterpret_cast<const uint64_t*>(item));
    case DType::Float32:
        return std::to_string(*reinterpret_cast<const float*>(item));
    case DType::Float64:
        return std::to_string(*reinterpret_cast<const double*>(item));
    }
    return "";
}

std::string Tensor::toString() const {
    // TODO : get a copy of this tensor on the cpu if not already there

    // TODO : handle adding ... for long tensors

    const ty::internal::ShapeStrides& layout = baseTensor_->getLayout();
    const std::array<size_t, internal::MAX_DIMS>& shape = layout.getShape();
    const size_t ndim = layout.getNDim();

    std::array<size_t, internal::MAX_DIMS> shapeProduct;
    size_t product = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        product *= shape[i];
        shapeProduct[i] = product;
    }

    const DType dtype = baseTensor_->getDType();

    size_t maxItemLength = 1;
    bool allItemsIntegral = true;
    for (auto it = begin(); it != end(); it++) {
        std::string itString = itemToStringCpu(*it, dtype);

        maxItemLength =
            (itString.size() > maxItemLength) ? itString.size() : maxItemLength;

        switch (dtype) {
        // TODO : Handle Float16 case
        case DType::Float32: {
            const float itFloat = *reinterpret_cast<const float*>(*it);
            allItemsIntegral &= std::trunc(itFloat) == itFloat;

        } break;
        case DType::Float64: {
            const double itDouble = *reinterpret_cast<const double*>(*it);
            allItemsIntegral &= std::trunc(itDouble) == itDouble;
        } break;
        default:
            break;
        }
    }

    if (!isIntegralType(dtype) && allItemsIntegral) {
        // Remove trailing zeros if all elements are whole numbers
        maxItemLength = (maxItemLength > 6) ? (maxItemLength - 6) : 1;
    }

    std::string resultStr = "";
    int index = 0;
    for (auto it = begin(); it != end(); it++, index++) {
        std::string itString = itemToStringCpu(*it, dtype);

        if (!isIntegralType(dtype) && allItemsIntegral) {
            size_t decimal_pos = itString.find('.');
            if (decimal_pos != std::string::npos) {
                itString.resize(decimal_pos + 1);
            } else {
                itString += ".";
            }
        }

        itString = std::string(maxItemLength - itString.size(), ' ') + itString;

        for (size_t i = 0; i < ndim; i++) {
            const int numBrackets = ndim - i;
            if (index % shapeProduct[i] == 0) {
                if (index != 0) {
                    resultStr += std::string(numBrackets, ']');
                    resultStr += numBrackets > 1 ? "\n\n" : "\n";
                }
                resultStr +=
                    std::string(i, ' ') + std::string(numBrackets, '[') + " ";
                break;
            }
        }

        resultStr += itString + " ";
    }

    resultStr += std::string(ndim, ']');

    return resultStr;
}
} // namespace ty