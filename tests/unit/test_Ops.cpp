#include "tityos/ty/ops/add.h"
#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <catch2/catch_all.hpp>

TEST_CASE("Tensor Addition", "[Operation][Pointwise]") {
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}),
                        std::vector<size_t>({2, 2}));
    ty::Tensor example2(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}),
                        std::vector<size_t>({2, 2}));

    auto result = ty::add(example1, example2);

    CHECK(*static_cast<float*>(result.at({0, 0})) == 2.0f);
    CHECK(*static_cast<float*>(result.at({0, 1})) == 4.0f);
    CHECK(*static_cast<float*>(result.at({1, 0})) == 6.0f);
    CHECK(*static_cast<float*>(result.at({1, 1})) == 8.0f);
}