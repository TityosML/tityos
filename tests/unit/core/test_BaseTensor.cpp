#include "tityos/ty/tensor/BaseTensor.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <catch2/catch_all.hpp>
#include <iostream>

TEST_CASE("BaseTensor creation", "[BaseTensor]") {
    REQUIRE_NOTHROW(ty::internal::BaseTensor(
        std::make_shared<ty::internal::TensorStorage>(16),
        ty::internal::ShapeStrides({2, 2}, 2), ty::DType::Int32));
    REQUIRE_NOTHROW(ty::internal::BaseTensor(
        std::make_shared<ty::internal::TensorStorage>(32),
        ty::internal::ShapeStrides({2, 3, 4}, 3)));
}

TEST_CASE("BaseTensor constructors", "[BaseTensor]") {
    ty::internal::BaseTensor originalExample1(
        std::make_shared<ty::internal::TensorStorage>(16),
        ty::internal::ShapeStrides({2, 2}, 2), ty::DType::Int32);
    ty::internal::BaseTensor originalExample2(
        std::make_shared<ty::internal::TensorStorage>(32),
        ty::internal::ShapeStrides({2, 3, 4}, 3));

    SECTION("Copy constructor") {
        REQUIRE_NOTHROW(ty::internal::BaseTensor(originalExample1));
        REQUIRE_NOTHROW(ty::internal::BaseTensor(originalExample2));
    };

    SECTION("Move constructor") {
        REQUIRE_NOTHROW(ty::internal::BaseTensor(std::move(originalExample1)));
        REQUIRE_NOTHROW(ty::internal::BaseTensor(std::move(originalExample2)));
    };
}

TEST_CASE("Accessing BaseTensor", "[BaseTensor]") {
    auto storage1 = std::make_shared<ty::internal::TensorStorage>(16);
    auto storage2 = std::make_shared<ty::internal::TensorStorage>(32);

    ty::internal::BaseTensor example1(
        storage1, ty::internal::ShapeStrides({2, 2}, 2), ty::DType::Int32);

    ty::internal::BaseTensor example2(storage2,
                                      ty::internal::ShapeStrides({2, 3, 4}, 3));

    CHECK(example1.getLayout() == ty::internal::ShapeStrides({2, 2}, 2));
    CHECK(example2.getLayout() == ty::internal::ShapeStrides({2, 3, 4}, 3));
    CHECK(example1.getTensorStorage() == storage1);
    CHECK(example2.getTensorStorage() == storage2);
    CHECK(example1.getDType() == ty::DType::Int32);
    CHECK(example2.getDType() == ty::DType::Float32);
}