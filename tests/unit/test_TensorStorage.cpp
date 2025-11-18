#include <catch2/catch_all.hpp>

#include "tityos/ty/tensor/TensorStorage.h"

TEST_CASE("TensorStorage can be allocated", "[TensorStorage]") {
    REQUIRE_NOTHROW([&]() { ty::internal::TensorStorage exampleArr(4 * sizeof(int)); });
}