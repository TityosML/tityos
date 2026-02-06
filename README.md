# Tityos

Tityos is a C++ deep learning library which supports SIMD parallelism via AVX for efficient CPU execution and CUDA for GPU acceleration


*This project is currently incomplete and as such may be subject to major changes*

#

#### Example
```c++
#include "tityos/ty/Tensors.h"
#include "tityos/ty/Ops.h"

int main() {
    ty::Tensor tensor1({1, 2, 3, 4, 5, 6}, {2, 3}, ty::DType::Int32);
    ty::Tensor tensor2({7, 8, 9, 10, 11, 12}, {3, 2}, ty::DType::Int32);

    auto result = ty::matmul(tensor1, tensor2);

    std::cout << result.toString() << "\n";
}
```

```terminal
[[ 58,  64],
 [139, 154]]
```

#

#### Installation
*Currently no install options for windows or mac*

##### Linux
With CUDA
```console
sudo cmake --install . -DTITYOS_BUILD_CUDA=ON
```
without CUDA
```console
sudo cmake --install .
```