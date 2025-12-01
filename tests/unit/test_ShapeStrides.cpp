#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <catch2/catch_all.hpp>

TEST_CASE("ShapeStrides creation", "[ShapeStrides]") {
    REQUIRE_NOTHROW(ty::internal::ShapeStrides({2, 2}, {2, 1}, 0, 2));
    REQUIRE_NOTHROW(ty::internal::ShapeStrides({2, 2}, ty::DType::Int32, 2));
    REQUIRE_NOTHROW(ty::internal::ShapeStrides({2, 2}, ty::DType::Int32, 2));
    REQUIRE_NOTHROW(
        ty::internal::ShapeStrides({2, 3, 4}, ty::DType::Float64, 3));
}

TEST_CASE("Accessing ShapeStrides and Computing Byte Index", "[ShapeStrides]") {
    ty::internal::ShapeStrides knownExample1({2, 2}, {2, 1}, 0, 2);
    ty::internal::ShapeStrides knownExample2({2}, {1}, 2, 1);
    ty::internal::ShapeStrides unknownExample1({2, 2}, ty::DType::Int32, 2);
    ty::internal::ShapeStrides unknownExample2({2, 3, 4}, ty::DType::Float64,
                                               3);

    CHECK(knownExample1.getNDim() == 2);
    CHECK(knownExample1.getShape() ==
          std::array<size_t, ty::internal::MAX_DIMS>{2, 2});
    CHECK(knownExample1.getStrides() ==
          std::array<size_t, ty::internal::MAX_DIMS>{2, 1});

    CHECK(knownExample2.getNDim() == 1);
    CHECK(knownExample2.getShape() ==
          std::array<size_t, ty::internal::MAX_DIMS>{2});
    CHECK(knownExample2.getStrides() ==
          std::array<size_t, ty::internal::MAX_DIMS>{1});

    CHECK(unknownExample1.getNDim() == 2);
    CHECK(unknownExample1.getShape() ==
          std::array<size_t, ty::internal::MAX_DIMS>{2, 2});
    CHECK(unknownExample1.getStrides() ==
          std::array<size_t, ty::internal::MAX_DIMS>{8, 4});

    CHECK(unknownExample2.getNDim() == 3);
    CHECK(unknownExample2.getShape() ==
          std::array<size_t, ty::internal::MAX_DIMS>{2, 3, 4});
    CHECK(unknownExample2.getStrides() ==
          std::array<size_t, ty::internal::MAX_DIMS>{96, 32, 8});

    size_t index1[1] = {1};
    size_t index2[2] = {1, 0};
    size_t index3[3] = {1, 2, 3};

    CHECK(knownExample1.computeByteIndex(index2) == 2);
    CHECK(knownExample2.computeByteIndex(index1) == 3);
    CHECK(unknownExample1.computeByteIndex(index2) == 8);
    CHECK(unknownExample2.computeByteIndex(index3) == 184);
}