#pragma once

#include "tityos/ty/export.h"
#include "tityos/ty/tensor/BaseTensor.h"
#include "tityos/ty/tensor/ShapeStrides.h"

namespace ty {
    namespace internal {
        class TensorIterator { 
          private:
            std::shared_ptr<BaseTensor> baseTensor_;
            std::array<size_t, MAX_DIMS> index_; 

            void* m_ptr;
            ShapeStrides layout_;
            size_t ndim_;
            std::array<size_t, MAX_DIMS> shape_;

            void incrementIndex() {
                for (size_t i = ndim_ - 1; i >= 0; i--)
                {
                    index_[i] += 1;
                    if (index_[i] < shape_[i]) {
                        break;
                    } else {
                        index_[i] = 0;
                    }
                }

                m_ptr = baseTensor_->at(index_);
            }
            
          public:
            TensorIterator(std::shared_ptr<BaseTensor> baseTensor, std::array<size_t, MAX_DIMS> startIndex) 
            : baseTensor_(baseTensor), index_(startIndex), layout_(baseTensor.getLayout()), m_ptr(baseTensor_->at(index_)), ndim_(layout.getNDim()), shape_(layout.getShape()) {}

            // TODO: use a template to add iterator tags

            reference operator*() const { return *m_ptr; }
            pointer operator->() { return m_ptr; }

            // Prefix increment
            Iterator& operator++() {
                incrementIndex();
                return *this; 
            }

            // Postfix increment
            Iterator operator++(int) { 
                TensorIterator tmp = *this;
                ++(*this);
                return tmp;
            } 

            friend bool operator== (const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
            friend bool operator!= (const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };   

            // TODO: move into BaseTensor as a struct if that is better for design
    };
  } // namespace internal
} // namespace ty