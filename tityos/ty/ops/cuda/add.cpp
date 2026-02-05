#include "tityos/ty/ops/cuda/add.h"

namespace ty {
namespace internal {
    BaseTensor backend::CUDABackend::add(const BaseTensor& tensor1,
                                         const BaseTensor& tensor2) {

        if (tensor1.getDType() != tensor2.getDType()) {
            throw std::invalid_argument("Types must match for addition");
        }

        BaseTensor result = internal::emptyLike(tensor1);

        DISPATCH_KERNEL_DTYPE_TABLE(
            kernelTable, launchAddKernel,
            (const TensorView&, const TensorView&, const TensorView&))

        kernelTable[static_cast<size_t>(tensor1.getDType())](result, tensor1,
                                                             tensor2);

        return result;
    }
} // namespace internal
} // namespace ty
