#include "tityos/ty/tensor/TensorStorage.h"

#include <catch2/catch_all.hpp>

TEST_CASE("TensorStorage can be allocated to CPU", "[TensorStorage]") {
    REQUIRE_NOTHROW(ty::internal::TensorStorage(4 * sizeof(int)));
    ty::internal::TensorStorage cpuStorage(4 * sizeof(int));
    SECTION("Begin pointer can be created") {
        REQUIRE_NOTHROW(cpuStorage.begin());
    }
    SECTION("End pointer can be created") {
        REQUIRE_NOTHROW(cpuStorage.end());
    }
}