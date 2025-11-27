#include "tityos/ty/tensor/Tensor.h"

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

std::string Tensor::toString() const {
    const std::array<size_t, internal::MAX_DIMS>& shape =
        baseTensor_->getLayout().getShape();
    const size_t ndim = baseTensor_->getLayout().getNDim();
    std::array<size_t, internal::MAX_DIMS> shapeProduct;
    size_t product = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        product *= shape[i];
        shapeProduct[i] = product;
    }

    // TODO : Add padding
    std::string str = "";
    int idx = 0;
    for (auto it = begin(); it != end(); it++, idx++) {
        for (size_t i = 0; i < ndim; i++) {
            if (idx % shapeProduct[i] == 0) {
                if (idx != 0) {
                    str += std::string(ndim - i, ']');
                    str += ndim - i > 1 ? "\n\n" : "\n";
                }
                str += std::string(ndim - i, '[') + " ";
                break;
            }
        }

        // TODO : Deal with different datatypes
        str += std::to_string(*static_cast<float*>(*it)) + " ";
    }

    str += std::string(ndim, ']');

    return str;
}
} // namespace ty