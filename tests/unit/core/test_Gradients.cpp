#include "tityos/ty/ops/add.h"
#include "tityos/ty/ops/bmm.h"
#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <catch2/catch_all.hpp>

TEST_CASE("Tensor Addition Gradients", "[Operation][Pointwise][Gradient]") {

    // Floats
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}),
                        std::vector<size_t>({2, 2}));
    ty::Tensor example2(std::vector<float>({5.0f, 6.0f, 7.0f, 8.0f}),
                        std::vector<size_t>({2, 2}));

    auto result1 = ty::add(example1, example2);
    result1.backward();

    auto example1Grad = ty::Tensor(example1.getGradTensor());
    auto example2Grad = ty::Tensor(example2.getGradTensor());

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            CHECK(example1Grad.elemAt<float>({i, j}) ==
                  example2.elemAt<float>({i, j}));

            CHECK(example2Grad.elemAt<float>({i, j}) ==
                  example1.elemAt<float>({i, j}));
        }
    }
}