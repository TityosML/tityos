#include "tityos/ty/ops/cuda/toCpu.h"

#include <cuda_runtime.h>

namespace ty {
namespace internal {
    BaseTensor backend::CUDABackend::toCpu(const BaseTensor& tensor) {
        // TODO: upgrade implementation so it doesnt require first making a
        // contiguous copy on the gpu

        Device cpuDevice(DeviceType::CPU);

        auto resultStorage =
            std::make_shared<TensorStorage>(tensor.getLogicalSize(), cpuDevice);

        auto contiguousTensor = contiguous(tensor);

        resultStorage->begin();
        cudaMemcpy(resultStorage->begin(),
                   contiguousTensor.getTensorStorage()->begin(),
                   contiguousTensor.getLogicalSize(), cudaMemcpyDeviceToHost);

        ShapeStrides resultLayout(tensor.getShape(), tensor.getNDim());
        BaseTensor result(resultStorage, resultLayout, tensor.getDType());

        return result;
    }
} // namespace internal
} // namespace ty
