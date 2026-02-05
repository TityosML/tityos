#include "tityos/ty/ops/cuda/bmm.h"

namespace ty {
namespace internal {
    BaseTensor backend::CUDABackend::bmm(const BaseTensor& batch1,
                                         const BaseTensor& batch2) {
        auto shape1 = batch1.getShape();
        auto shape2 = batch2.getShape();

        TensorShape resultShape = {shape1[0], shape1[1], shape2[2]};
        BaseTensor result = internal::empty(resultShape, 3, batch1.getDType(),
                                            batch1.getDevice());

        DISPATCH_KERNEL_DTYPE_TABLE(
            kernelTable, launchBMMKernel,
            (const BaseTensor&, const BaseTensor&, BaseTensor&))

        kernelTable[static_cast<size_t>(tensor1.getDType())](result, tensor1,
                                                             tensor2);

        return result;
    }
} // namespace internal
} // namespace ty
