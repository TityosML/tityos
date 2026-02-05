#include "tityos/ty/ops/cuda/contiguous.h"

namespace ty {
namespace internal {
    BaseTensor backend::CUDABackend::contiguous(const BaseTensor& tensor) {
        BaseTensor result = internal::emptyLike(tensor);

        DISPATCH_KERNEL_DTYPE_TABLE(kernelTable, launchContiguousKernel,
                                    (const BaseTensor&, BaseTensor&))

        kernelTable[static_cast<size_t>(tensor1.getDType())](result, tensor);

        return result;
    }
} // namespace internal
} // namespace ty
