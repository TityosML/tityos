#include "tityos/ty/tensor/Tensor.h"

namespace ty {
Tensor::Tensor(const Tensor& other) : baseTensor_(other.baseTensor_) {}

Tensor::Tensor(Tensor&& other) noexcept
    : baseTensor_(std::move(other.baseTensor_)) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }

    baseTensor_ = other.baseTensor_;

    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    baseTensor_ = std::move(other.baseTensor_);

    return *this;
}

Tensor Tensor::copy() const {
    return Tensor(std::make_shared<internal::BaseTensor>(baseTensor_->copy()));
}

Device Tensor::getDevice() const {
    return baseTensor_->getTensorStorage()->getDevice();
}

DType Tensor::getDType() const {
    return baseTensor_->getDType();
}

const std::array<size_t, TY_MAX_DIMS>& Tensor::getShape() const {
    return baseTensor_->getLayout().getShape();
}

size_t Tensor::getSize() const {
    size_t tensorSize = 1;
    const std::array<size_t, TY_MAX_DIMS>& shape = getShape();
    for (size_t i = 0; i < baseTensor_->getLayout().getNDim(); i++) {
        tensorSize *= shape[i];
    }
    return tensorSize;
}

size_t Tensor::getNDim() const {
    return baseTensor_->getLayout().getNDim();
}

std::shared_ptr<internal::BaseTensor> Tensor::getBaseTensor() const {
    return baseTensor_;
}

void* Tensor::at(const size_t* indexStart, const size_t indexSize) const {
    if (indexSize != baseTensor_->getLayout().getNDim()) {
        throw std::invalid_argument(
            "Index size mismatch: expected " +
            std::to_string(baseTensor_->getLayout().getNDim()) + ", got " +
            std::to_string(indexSize));
    }

    const std::array<size_t, TY_MAX_DIMS>& shape =
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

bool Tensor::isContiguous() const {
    return baseTensor_->isContiguous();
}

std::string Tensor::toString() const {
    const int ElideAfter = 0; // currently 0 for testing, should be 1000
    const int ElidedElementsPrinted = 3;
    const int DecimalsDisplayed = 6;

    // TODO : get a copy of this tensor on the cpu if not already there
    auto b = internal::backend::getBackend(getDevice().type());
    const auto baseTensorCpu = b->toCpu(*baseTensor_);

    const ty::internal::ShapeStrides& layout = baseTensor_->getLayout();
    const TensorShape& shape = layout.getShape();
    const DType dtype = getDType();
    const size_t ndim = layout.getNDim();
    const size_t tensorSize = getSize();

    std::ostringstream resultStream;

    // returns (finished iterating, num brackets to place, has elided)
    auto nextVisibleIndex =
        [=](size_t& linearIndex) -> std::tuple<bool, size_t, bool> {
        auto index = layout.linearToTensorIndex(linearIndex);

        for (int i = ndim - 1; i >= 0; i--) {
            index[i]++;

            size_t dimSize = shape[i];
            bool isElided =
                tensorSize > ElideAfter && dimSize > 2 * ElidedElementsPrinted;

            if (isElided && index[i] == ElidedElementsPrinted) {
                index[i] = dimSize - ElidedElementsPrinted;
                linearIndex = layout.tensorToLinearIndex(index);
                return {true, ndim - i - 1, true};
            }
            if (index[i] < dimSize) {
                linearIndex = layout.tensorToLinearIndex(index);
                return {true, ndim - i - 1, false};
            }

            index[i] = 0;
        }
        linearIndex = layout.tensorToLinearIndex(index);
        return {false, ndim, false};
    };

    size_t maxItemLength = 1;
    bool allVisibleIntegral = true;

    {
        // First pass, finds length of items to determine padding
        bool active = true;
        auto it = baseTensorCpu.begin();
        auto idx = it.getIndex();

        while (active) {
            it.jumpToIndex(idx);

            std::string itString = itemToStringCpu(*it, dtype);

            maxItemLength = (itString.size() > maxItemLength) ? itString.size()
                                                              : maxItemLength;

            switch (dtype) {
            // TODO : Handle Float16 case
            case DType::Float32: {
                const float itFloat = *reinterpret_cast<const float*>(*it);
                allVisibleIntegral &= std::trunc(itFloat) == itFloat;

            } break;
            case DType::Float64: {
                const double itDouble = *reinterpret_cast<const double*>(*it);
                allVisibleIntegral &= std::trunc(itDouble) == itDouble;
            } break;
            default:
                break;
            }

            active = std::get<0>(nextVisibleIndex(idx));
        }
    }

    if (!isIntegralType(dtype) && allVisibleIntegral) {
        // Remove trailing zeros if all elements are whole numbers
        maxItemLength = (maxItemLength > DecimalsDisplayed)
                            ? (maxItemLength - DecimalsDisplayed)
                            : 1;
    }

    {
        // Second pass, constructs the output string
        auto it = baseTensorCpu.begin();
        auto index = it.getIndex();

        bool active = true;
        size_t numBrackets = 0;
        bool hasElided = false;

        resultStream << std::string(ndim, '[') << " ";

        while (active) {
            it.jumpToIndex(index);

            if (numBrackets > 0) {
                resultStream << std::string(numBrackets, ']') << "\n";

                if (hasElided) {
                    if (numBrackets > 1) {
                        resultStream << "\n"
                                     << std::string(ndim - numBrackets, ' ')
                                     << "...\n\n";
                    } else {
                        resultStream << std::string(ndim - numBrackets, ' ')
                                     << "...\n";
                    }
                } else if (numBrackets > 1) {
                    resultStream << "\n";
                }

                resultStream << std::string(ndim - numBrackets, ' ')
                             << std::string(numBrackets, '[') << " ";
            } else if (hasElided) {
                resultStream << "... ";
            }

            std::string itString = itemToStringCpu(*it, dtype);

            if (!isIntegralType(dtype) && allVisibleIntegral) {
                size_t decimal_pos = itString.find('.');
                // Assumes that a '.' is always found
                itString.resize(decimal_pos + 1);
            }

            itString =
                std::string(maxItemLength - itString.size(), ' ') + itString;

            resultStream << itString << " ";

            std::tie(active, numBrackets, hasElided) = nextVisibleIndex(index);
        }
    }

    resultStream << std::string(ndim, ']');

    return resultStream.str();
}
} // namespace ty