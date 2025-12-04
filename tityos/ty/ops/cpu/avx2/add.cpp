#include "tityos/ty/ops/cpu/avx2/add.h"

namespace ty {
namespace internal {
    void internalAddAvx2(Tensor& result, const Tensor& tensor1, const Tensor& tensor2) {
        auto resultIt = result.begin();
        auto it1 = tensor1.begin();
        auto it2 = tensor2.begin();

        // TODO: Make this work with all datatypes
        while (resultIt != result.end()) {
            *(reinterpret_cast<float*>(*resultIt)) = *(reinterpret_cast<float*>(*it1)) + *(reinterpret_cast<float*>(*it2));
            resultIt++;
            it1++;
            it2++;
        }
    }
} // namespace internal
} // namespace ty