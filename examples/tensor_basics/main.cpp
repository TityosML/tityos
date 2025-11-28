#include "tityos/ty/tensor/Tensor.h"

#include <iostream>

int main() {
    ty::Tensor example1({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    std::cout << example1.toString() << std::endl;
}