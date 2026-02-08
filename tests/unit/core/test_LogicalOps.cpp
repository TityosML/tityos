#include "tityos/ty/ops/contiguous.h"
#include "tityos/ty/ops/expand.h"
#include "tityos/ty/ops/reshape.h"
#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <catch2/catch_all.hpp>

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

TEST_CASE("Tensor CPU contiguous", "[Operation]") {
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}),
                        std::vector<size_t>({2, 2}));

    ty::Tensor exampleSliced(std::make_shared<ty::internal::BaseTensor>(
        example1.getBaseTensor()->slice(1, 0, 1)));

    auto result = ty::contiguous(exampleSliced);

    CHECK(exampleSliced.isContiguous() == false);
    CHECK(result.isContiguous() == true);

    CHECK(exampleSliced.elemAt<float>({0, 0}) == result.elemAt<float>({0, 0}));
    CHECK(exampleSliced.elemAt<float>({1, 0}) == result.elemAt<float>({1, 0}));

    CHECK(exampleSliced.elemAt<float>({0, 0}) == 1.0f);
}

TEST_CASE("Tensor Reshape", "[Operation][View]") {
    ty::Tensor t(std::vector<float>({1.f, 2.f, 3.f, 4.f}),
                 std::vector<size_t>({2, 2}));

    auto reshaped = ty::reshape(t, {4}, 1);

    CHECK(reshaped.elemAt<float>({0}) == 1.f);
    CHECK(reshaped.elemAt<float>({1}) == 2.f);
    CHECK(reshaped.elemAt<float>({2}) == 3.f);
    CHECK(reshaped.elemAt<float>({3}) == 4.f);

    CHECK(reshaped.isContiguous() == true);
}

TEST_CASE("Tensor View", "[Operation][View]") {
    ty::Tensor t(std::vector<float>({1.f, 2.f, 3.f, 4.f}),
                 std::vector<size_t>({2, 2}));

    auto viewed = ty::view(t, {4}, 1);

    CHECK(viewed.elemAt<float>({0}) == 1.f);
    CHECK(viewed.elemAt<float>({1}) == 2.f);
    CHECK(viewed.elemAt<float>({2}) == 3.f);
    CHECK(viewed.elemAt<float>({3}) == 4.f);

    CHECK(viewed.getBaseTensor()->getTensorStorage() ==
          t.getBaseTensor()->getTensorStorage());
}

TEST_CASE("Tensor Reshape forces contiguous", "[Operation][View]") {
    ty::Tensor t(std::vector<float>({1.f, 2.f, 3.f, 4.f}),
                 std::vector<size_t>({2, 2}));

    ty::Tensor sliced(std::make_shared<ty::internal::BaseTensor>(
        t.getBaseTensor()->slice(1, 0, 1)));

    REQUIRE(sliced.isContiguous() == false);

    auto reshaped = ty::reshape(sliced, {2}, 1);

    CHECK(reshaped.isContiguous() == true);
    CHECK(reshaped.elemAt<float>({0}) == 1.f);
    CHECK(reshaped.elemAt<float>({1}) == 3.f);
}