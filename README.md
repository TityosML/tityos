# Tityos

Tityos is a C++ deep learning library which supports SIMD parallelism via AVX for efficient CPU execution and CUDA for GPU acceleration


*This project is currently incomplete and as such may be subject to major changes*

#

#### Example
```c++
#include "tityos/ty/tensor/Tensor.h"
#include "tityos/ty/ops/bmm.h"

int main() {
    constexpr auto device = {ty::DeviceType::CUDA, 0};
    ty::Tensor tensor1({1, 2, 3, 4, 5, 6}, {1, 2, 3}, device, ty::DType::Int32);
    ty::Tensor tensor2({7, 8, 9, 10, 11, 12}, {1, 3, 2}, device, ty::DType::Int32);

    auto result = ty::bmm(tensor1, tensor2);

    std::cout << result.toString() << "\n";
}
```

```terminal
[[[ 58,  64],
  [139, 154]]]
```

#

#### Installation
*Currently no install options for windows or mac*

##### Linux
With CUDA
```console
cmake -S . -B build -DTITYOS_BUILD_CUDA=ON
cmake --build build --target tityos_cpu
cmake --build build --target tityos_cuda
sudo cmake --install build -DTITYOS_BUILD_CUDA=ON
```
without CUDA
```console
cmake -S . -B build
cmake --build build --target tityos_cpu
sudo cmake --install build
```
