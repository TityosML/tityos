#include "tityos/ty/ops/add.h"
#include "tityos/ty/ops/expand.h"
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

    // Floats with broadcasting
    ty::Tensor example3(
        std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}),
        std::vector<size_t>({3, 2}));

    ty::Tensor example4(std::vector<float>({10.0f, 20.0f}),
                        std::vector<size_t>({2}));

    auto result2 = ty::add(example3, example4);

    CHECK(result2.elemAt<float>({0, 0}) == 11.0f);
    CHECK(result2.elemAt<float>({0, 1}) == 22.0f);

    CHECK(result2.elemAt<float>({1, 0}) == 13.0f);
    CHECK(result2.elemAt<float>({1, 1}) == 24.0f);

    CHECK(result2.elemAt<float>({2, 0}) == 15.0f);
    CHECK(result2.elemAt<float>({2, 1}) == 26.0f);

    // Int32
    ty::Tensor example5(std::vector<int>({1, 2, 3, 4}),
                        std::vector<size_t>({2, 2}), {ty::DeviceType::CPU, 0},
                        ty::DType::Int32);
    ty::Tensor example6(std::vector<int>({5, 6, 7, 8}),
                        std::vector<size_t>({2, 2}), {ty::DeviceType::CPU, 0},
                        ty::DType::Int32);

    auto result3 = ty::add(example5, example6);

    CHECK(result3.elemAt<int>({0, 0}) == 6);
    CHECK(result3.elemAt<int>({0, 1}) == 8);
    CHECK(result3.elemAt<int>({1, 0}) == 10);
    CHECK(result3.elemAt<int>({1, 1}) == 12);
}

TEST_CASE("Tensor Expand", "[Operation][Pointwise]") {
    // Floats
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}),
                        std::vector<size_t>({2, 2}));

    auto result1 = ty::expand(example1, {4, 2, 2});

    CHECK(result1.elemAt<float>({0, 0, 0}) == 1.0f);
    CHECK(result1.elemAt<float>({0, 0, 1}) == 2.0f);
    CHECK(result1.elemAt<float>({0, 1, 0}) == 3.0f);
    CHECK(result1.elemAt<float>({0, 1, 1}) == 4.0f);

    CHECK(result1.elemAt<float>({1, 0, 0}) == 1.0f);
    CHECK(result1.elemAt<float>({2, 0, 1}) == 2.0f);
    CHECK(result1.elemAt<float>({3, 1, 0}) == 3.0f);
    CHECK(result1.elemAt<float>({3, 1, 1}) == 4.0f);
}
