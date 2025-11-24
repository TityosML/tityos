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

            
            std::array<size_t, MAX_DIMS> endIndex() const {
                // Inclusive end index
                std::array<size_t, MAX_DIMS> endIdx = {};
                for (size_t i = 0; i < layout_.getNDim(); i++){
                    endIdx[i] = layout_.getShape()[i]-1;
                }
                return endIdx;
            }

          
          public:
            BaseTensor(std::shared_ptr<TensorStorage> data, const ShapeStrides &layout)
                : tensorStorage_(std::move(data)), layout_(layout) {}

            void *at(const size_t *indexStart) const {
                size_t byteOffset = layout_.computeByteIndex(indexStart);
                return tensorStorage_->at(byteOffset);
            }

            const ShapeStrides &getLayout() const {
                return layout_;
            }
            const std::shared_ptr<TensorStorage> &getByteArray() const {
                return tensorStorage_;
            }

            struct Iterator {
                Iterator(const BaseTensor &baseTensor, const std::array<size_t, MAX_DIMS> &startIndex) 
                : baseTensor_(baseTensor), index_(startIndex), ptr_(baseTensor.at(startIndex.data())) {}
                
                void* operator->() { return ptr_; }
                void* operator*() { return ptr_; }
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
                const BaseTensor &baseTensor_;
                std::array<size_t, MAX_DIMS> index_; 

                void* ptr_;

                void incrementIndex() {
                    for (int i = baseTensor_.getLayout().getNDim() - 1; i >= 0; i--) {
                        index_[i]++;
                        if (index_[i] < baseTensor_.getLayout().getShape()[i]) {
                            break;
                        } else {
                            index_[i] = 0;
                        }
                    }

                    ptr_ = baseTensor_.at(index_.data());
                }
            }; 

            Iterator begin() { return Iterator(*this, std::array<size_t, MAX_DIMS> {});}
            Iterator end() { return Iterator(*this, endIndex())++;}

            Iterator begin() const { return Iterator(*this, std::array<size_t, MAX_DIMS> {});}
            Iterator end() const { return Iterator(*this, endIndex())++;}
        };
    } // namespace internal
} // namespace ty