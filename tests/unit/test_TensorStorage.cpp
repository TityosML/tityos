#include <catch2/catch_all.hpp>

#include "tityos/ty/tensor/TensorStorage.h"

TEST_CASE("TensorStorage can be allocated to CPU", "[TensorStorage]") {
    REQUIRE_NOTHROW([&]() { ty::internal::TensorStorage exampleArr(4 * sizeof(int)); });
}

TEST_CASE("TensorStorage can be allocated to GPU", "[TensorStorage]") {
    REQUIRE_NOTHROW([&]() { ty::internal::TensorStorage exampleArr(4 * sizeof(int), {ty::DeviceType::CUDA, 0}); });
}