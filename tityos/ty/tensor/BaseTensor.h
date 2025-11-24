#pragma once

#include <memory>

#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"
#include "tityos/ty/tensor/TensorStorage.h"

namespace ty {
    namespace internal {
        class BaseTensor {
          private:
            DType dtype_;

            std::shared_ptr<TensorStorage> tensorStorage_;
            ShapeStrides layout_;
          
          public:
            BaseTensor(std::shared_ptr<TensorStorage> data, const ShapeStrides &layout)
                : tensorStorage_(std::move(data)), layout_(layout) {}

            void *at(const std::array<size_t, MAX_DIMS> &index) const {
                size_t byteOffset = layout_.computeByteIndex(index);
                return tensorStorage_->at(byteOffset);
            }

            const ShapeStrides &getLayout() const {
                return layout_;
            }
            const std::shared_ptr<TensorStorage> &getByteArray() const {
                return tensorStorage_;
            }

            struct Iterator {
                Iterator(BaseTensor &baseTensor, std::array<size_t, MAX_DIMS> startIndex) 
                : baseTensor_(baseTensor), index_(startIndex), ptr_((&baseTensor)->at(startIndex)) {}
                
                void* operator->() { return ptr_; }
                // Prefix increment
                Iterator& operator++() {
                    incrementIndex();
                    return *this; 
                }

                // Postfix increment
                Iterator operator++(int) { 
                    Iterator tmp = *this;
                    ++(*this);
                    return tmp;
                } 

                friend bool operator== (const Iterator& a, const Iterator& b) { return a.ptr_ == b.ptr_; };
                friend bool operator!= (const Iterator& a, const Iterator& b) { return a.ptr_ != b.ptr_; };   

              private:
                BaseTensor &baseTensor_;
                std::array<size_t, MAX_DIMS> index_; 

                void* ptr_;

                void incrementIndex() {
                    for (size_t i = baseTensor_.getLayout().getNDim(); i > 0; i--) {
                        index_[i-1] += 1;
                        if (index_[i-1] < baseTensor_.getLayout().getShape()[i-1]) {
                            break;
                        } else {
                            index_[i-1] = 0;
                        }
                    }

                    ptr_ = (&baseTensor_)->at(index_);
                }
            }; 
        };
    } // namespace internal
} // namespace ty