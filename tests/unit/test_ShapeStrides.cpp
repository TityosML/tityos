#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <catch2/catch_all.hpp>

TEST_CASE("ShapeStrides creation", "[ShapeStrides]") {
    REQUIRE_NOTHROW(ty::internal::ShapeStrides({2, 2}, {2, 1}, 0, 2));
    REQUIRE_NOTHROW(ty::internal::ShapeStrides({2, 2}, 2));
    REQUIRE_NOTHROW(ty::internal::ShapeStrides({2, 2}, 2));
    REQUIRE_NOTHROW(ty::internal::ShapeStrides({2, 3, 4}, 3));
}

TEST_CASE("Accessing ShapeStrides and Computing Byte Index", "[ShapeStrides]") {
    ty::internal::ShapeStrides unknownExample1({2, 2}, 2);
    ty::internal::ShapeStrides unknownExample2({2, 3, 4}, 3);

    CHECK(unknownExample1.getNDim() == 2);
    CHECK(unknownExample1.getShape() ==
          std::array<size_t, ty::internal::MAX_DIMS>{2, 2});
    CHECK(unknownExample1.getStrides() ==
          std::array<size_t, ty::internal::MAX_DIMS>{2, 1});

    CHECK(unknownExample2.getNDim() == 3);
    CHECK(unknownExample2.getShape() ==
          std::array<size_t, ty::internal::MAX_DIMS>{2, 3, 4});
    CHECK(unknownExample2.getStrides() ==
          std::array<size_t, ty::internal::MAX_DIMS>{12, 4, 1});

    size_t index1[1] = {1};
    size_t index2[2] = {1, 0};
    size_t index3[3] = {1, 2, 3};

    CHECK(unknownExample1.computeByteIndex(index2, ty::DType::Int32) == 8);
    CHECK(unknownExample2.computeByteIndex(index3, ty::DType::Float64) == 184);
}

TEST_CASE("ShapeStrides Linear and Tensor index conversion idempotence", "[ShapeStrides]") {
    ty::internal::ShapeStrides example1({2, 3}, 2);

    CHECK(example1.tensorToLinearIndex(example1.linearToTensorIndex(1)) == 1);
    CHECK(example1.tensorToLinearIndex(example1.linearToTensorIndex(2)) == 2);
    CHECK(example1.tensorToLinearIndex(example1.linearToTensorIndex(3)) == 3);
}