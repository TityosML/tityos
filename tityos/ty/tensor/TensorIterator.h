#pragma once

#include "tityos/ty/export.h"
#include "tityos/ty/tensor/BaseTensor.h"
#include "tityos/ty/tensor/ShapeStrides.h"

namespace ty {
    namespace internal {
        template <typename T>
        class TensorIterator { 
          private:
            std::shared_ptr<BaseTensor> baseTensor_;
            std::array<size_t, MAX_DIMS> index_; 

            pointer m_ptr;
            void incrementIndex() {
                for (size_t i = layout.getNDim(); i > 0; i--)
                {
                    size_t idx = i - 1;
                    index_[idx] += 1;
                    if (index_[idx] < layout.getShape()[idx]) {
                        break;
                    } else {
                        index_[idx] = 0;
                    }
                }

                m_ptr = baseTensor_->at(index_);
            }
            
          public:
            TensorIterator(std::shared_ptr<BaseTensor> baseTensor, std::array<size_t, MAX_DIMS> startIndex) 
            : baseTensor_(baseTensor), index_(startIndex), m_ptr(baseTensor_->at(index_)) {}

            // TODO: replace float with a generic type
            using iterator_category = std::forward_iterator_tag;
            using value_type        = T;
            using pointer           = T*;  
            using reference         = T&;  

            reference operator*() const { return *m_ptr; }
            pointer operator->() { return m_ptr; }
            // Prefix increment
            TensorIterator& operator++() {
                incrementIndex();
                return *this; 
            }

            // Postfix increment
            TensorIterator operator++(int) { 
                TensorIterator tmp = *this;
                ++(*this);
                return tmp;
            } 

            bool operator== (const TensorIterator& a, const TensorIterator& b) { return a.m_ptr == b.m_ptr; };
            bool operator!= (const TensorIterator& a, const TensorIterator& b) { return a.m_ptr != b.m_ptr; };   

            // TODO: move into BaseTensor as a struct if that is better for design

    };
  } // namespace internal
} // namespace ty