#include "tityos/ty/ops/cuda/bmm.h"

#include <cuda_runtime.h>

namespace ty {
namespace internal {

    constexpr int Tx = 16;
    constexpr int Ty = 16;

    template <typename T>
    __global__ void
    bmmCudaKernel(int batches, int M, int N, int K, int sx1, int sy1, int sz1,
                  int sx2, int sy2, int sz2, int sxRes, int syRes, int szRes,
                  CudaTensorView<T> tensorView1, CudaTensorView<T> tensorView2,
                  CudaTensorView<T> resultTensorView, float alpha, float beta) {
        const int batch = blockIdx.z;
        // TODO: figure out coalesce
        const int x = blockIdx.x * Tx + threadIdx.x;
        const int y = blockIdx.y * Ty + threadIdx.y;

        if (x < M && y < N && batch < batches) {
            T tmp = 0;

            for (int i = 0; i < K; i++) {
                tmp += tensorView1.at3d(i, y, batch, sx1, sy1, sz1) *
                       tensorView2.at3d(x, i, batch, sx2, sy2, sz2);
            }
            // C = α*(A@B)+β*C
            resultTensorView.at3d(x, y, batch, sxRes, syRes, szRes) =
                alpha * tmp +
                beta * resultTensorView.at3d(x, y, batch, sxRes, syRes, szRes);
        };
    };

    template <typename T>
    void launchBMMKernel(BaseTensor& result, const BaseTensor& batch1,
                         const BaseTensor& batch2) {

        // convert all tensors to tensorviews
        CudaTensorView<T> tensorView1 = buildCudaTensorView<T>(batch1);
        CudaTensorView<T> tensorView2 = buildCudaTensorView<T>(batch2);
        CudaTensorView<T> resultTensorView = buildCudaTensorView<T>(result);

        ty::internal::TensorShape resultShape = result.getShape();
        int batches = resultShape[0];
        int cols = resultShape[1];
        int rows = resultShape[2];

        dim3 block(Tx, Ty);
        dim3 grid((cols + Tx - 1) / Tx, (rows + Ty - 1) / Tx, batches);

        // launch the bmm kernel
        // TODO: add parameters for alpha and beta
        bmmCudaKernel<T><<<grid, block>>>(
            batches, rows, cols, tensorView1.shape[2], tensorView1.strides[2],
            tensorView1.strides[1], tensorView1.strides[0],
            tensorView2.strides[2], tensorView2.strides[1],
            tensorView2.strides[0], resultTensorView.strides[2],
            resultTensorView.strides[1], resultTensorView.strides[0],
            tensorView1, tensorView2, resultTensorView, 1.0f, 0.0f);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // force compiler to compile each type
    template void launchBMMKernel<int8_t>(BaseTensor& result,
                                          const BaseTensor& batch1,
                                          const BaseTensor& batch2);
    template void launchBMMKernel<uint8_t>(BaseTensor& result,
                                           const BaseTensor& batch1,
                                           const BaseTensor& batch2);
    template void launchBMMKernel<int16_t>(BaseTensor& result,
                                           const BaseTensor& batch1,
                                           const BaseTensor& batch2);
    template void launchBMMKernel<uint16_t>(BaseTensor& result,
                                            const BaseTensor& batch1,
                                            const BaseTensor& batch2);
    template void launchBMMKernel<int32_t>(BaseTensor& result,
                                           const BaseTensor& batch1,
                                           const BaseTensor& batch2);
    template void launchBMMKernel<uint32_t>(BaseTensor& result,
                                            const BaseTensor& batch1,
                                            const BaseTensor& batch2);
    template void launchBMMKernel<int64_t>(BaseTensor& result,
                                           const BaseTensor& batch1,
                                           const BaseTensor& batch2);
    template void launchBMMKernel<uint64_t>(BaseTensor& result,
                                            const BaseTensor& batch1,
                                            const BaseTensor& batch2);
    template void launchBMMKernel<float>(BaseTensor& result,
                                         const BaseTensor& batch1,
                                         const BaseTensor& batch2);
    template void launchBMMKernel<double>(BaseTensor& result,
                                          const BaseTensor& batch1,
                                          const BaseTensor& batch2);

} // namespace internal
} // namespace ty
