#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <catch2/catch_all.hpp>

TEST_CASE("ShapeStrides creation", "[ShapeStrides]") {
    REQUIRE_NOTHROW(ty::internal::ShapeStrides({2, 2}, {2, 1}, 0, 2));
    REQUIRE_NOTHROW(ty::internal::ShapeStrides({2, 2}, 2));
    REQUIRE_NOTHROW(ty::internal::ShapeStrides({2, 3, 4}, 3));
}

TEST_CASE("Accessing ShapeStrides and Computing Byte Index", "[ShapeStrides]") {
    ty::internal::ShapeStrides unknownExample1({2, 2}, 2);
    ty::internal::ShapeStrides unknownExample2({2, 3, 4}, 3);

    CHECK(unknownExample1.getNDim() == 2);
    CHECK(unknownExample1.getShape() == ty::TensorShape{2, 2});
    CHECK(unknownExample1.getStrides() == ty::TensorStrides{2, 1});

    CHECK(unknownExample2.getNDim() == 3);
    CHECK(unknownExample2.getShape() == ty::TensorShape{2, 3, 4});
    CHECK(unknownExample2.getStrides() == ty::TensorStrides{12, 4, 1});

    size_t index1[1] = {1};
    size_t index2[2] = {1, 0};
    size_t index3[3] = {1, 2, 3};

    CHECK(unknownExample1.computeByteIndex(index2, ty::DType::Int32) == 8);
    CHECK(unknownExample2.computeByteIndex(index3, ty::DType::Float64) == 184);
}

TEST_CASE("ShapeStrides Linear and Tensor index conversion idempotence",
          "[ShapeStrides]") {
    ty::internal::ShapeStrides example1({2, 3}, 2);

    CHECK(example1.tensorToLinearIndex(example1.linearToTensorIndex(1)) == 1);
    CHECK(example1.tensorToLinearIndex(example1.linearToTensorIndex(2)) == 2);
    CHECK(example1.tensorToLinearIndex(example1.linearToTensorIndex(3)) == 3);
}

TEST_CASE("ShapeStrides Slicing", "[ShapeStrides]") {
    ty::internal::ShapeStrides example1D({10}, {1}, 0, 1);
    ty::internal::ShapeStrides example2D({3, 4}, {4, 1}, 0, 2);

    CHECK(example1D.slice(0) == example1D);
    CHECK(example1D.slice(0, 6) == ty::internal::ShapeStrides({4}, {1}, 6, 1));
    CHECK(example1D.slice(0, 1, 8, 3) ==
          ty::internal::ShapeStrides({3}, {3}, 1, 1));
    CHECK(example1D.slice(0, std::nullopt, 4) ==
          ty::internal::ShapeStrides({4}, {1}, 0, 1));
    CHECK(example1D.slice(0, std::nullopt, std::nullopt, 2) ==
          ty::internal::ShapeStrides({5}, {2}, 0, 1));
    CHECK(example1D.slice(0, std::nullopt, std::nullopt, -1) ==
          ty::internal::ShapeStrides({10}, {-1}, 9, 1));

    CHECK(example2D.slice(0) == example2D);
    CHECK(example2D.slice(1) == example2D);
    CHECK(example2D.slice(0, 0, 2) ==
          ty::internal::ShapeStrides({2, 4}, {4, 1}, 0, 2));
    CHECK(example2D.slice(0, 0, 2).slice(1, 1, 3) ==
          ty::internal::ShapeStrides({2, 2}, {4, 1}, 1, 2));
    CHECK(example2D.slice(0).slice(1, -2) ==
          ty::internal::ShapeStrides({3, 2}, {4, 1}, 2, 2));
    CHECK(example2D.slice(0, 0, 1).slice(1, 1) ==
          ty::internal::ShapeStrides({1, 3}, {4, 1}, 1, 2));
    CHECK(example2D.slice(0, std::nullopt, std::nullopt, 2).slice(1) ==
          ty::internal::ShapeStrides({2, 4}, {8, 1}, 0, 2));
}