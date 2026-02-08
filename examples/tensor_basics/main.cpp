#include "tityos/ty/ops/bmm.h"
#include "tityos/ty/tensor/Tensor.h"

#include <iostream>

int main() {
    const ty::Device device{ty::DeviceType::CPU, 0};
    ty::Tensor tensor1({1, 2, 3, 4, 5, 6}, {1, 2, 3}, device, ty::DType::Int32);
    ty::Tensor tensor2({7, 8, 9, 10, 11, 12}, {1, 3, 2}, device,
                       ty::DType::Int32);

    auto result = ty::bmm(tensor1, tensor2);

    std::cout << result.toString() << "\n";
}