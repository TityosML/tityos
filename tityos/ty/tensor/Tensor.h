#pragma once

#include <string.h>

#include "tityos/ty/tensor/BaseTensor.h"


namespace ty {
    namespace internal {
        class Tensor {
          private: 
            std::shared_ptr<BaseTensor> baseTensor_;

          public:
            Tensor(std::shared_ptr<TensorStorage> data, const ShapeStrides &layout)
                : baseTensor_(BaseTensor(data, &layout)) {}
            
            const std::string toString const {
                // TODO
            }
        }
    }
}