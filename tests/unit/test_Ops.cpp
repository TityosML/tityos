#include "tityos/ty/ops/add.h"
#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <catch2/catch_all.hpp>

TEST_CASE("Tensor Addition", "[Operation][Pointwise]") {

    // Floats
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}),
                        std::vector<size_t>({2, 2}));
    ty::Tensor example2(std::vector<float>({5.0f, 6.0f, 7.0f, 8.0f}),
                        std::vector<size_t>({2, 2}));

    auto result1 = ty::add(example1, example2);

    CHECK(result1.elemAt<float>({0, 0}) == 6.0f);
    CHECK(result1.elemAt<float>({0, 1}) == 8.0f);
    CHECK(result1.elemAt<float>({1, 0}) == 10.0f);
    CHECK(result1.elemAt<float>({1, 1}) == 12.0f);

    // Int32
    ty::Tensor example3(std::vector<int>({1, 2, 3, 4}),
                        std::vector<size_t>({2, 2}), {ty::DeviceType::CPU, 0},
                        ty::DType::Int32);
    ty::Tensor example4(std::vector<int>({5, 6, 7, 8}),
                        std::vector<size_t>({2, 2}), {ty::DeviceType::CPU, 0},
                        ty::DType::Int32);

    auto result2 = ty::add(example3, example4);

    CHECK(result2.elemAt<int>({0, 0}) == 6);
    CHECK(result2.elemAt<int>({0, 1}) == 8);
    CHECK(result2.elemAt<int>({1, 0}) == 10);
    CHECK(result2.elemAt<int>({1, 1}) == 12);
}