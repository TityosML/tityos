#include "tityos/ty/tensor/TensorStorage.h"

#include <catch2/catch_all.hpp>

TEST_CASE("TensorStorage can be allocated to GPU with CUDA", "[TensorStorage]") {
    REQUIRE_NOTHROW(ty::internal::TensorStorage(4 * sizeof(int), {ty::DeviceType::CUDA, 0}));
    ty::internal::TensorStorage gpuStorage(4 * sizeof(int), {ty::DeviceType::CUDA, 0});
    SECTION("Begin pointer can be created") {
        REQUIRE_NOTHROW(gpuStorage.begin());
    }
    SECTION("End pointer can be created") {
        REQUIRE_NOTHROW(gpuStorage.end());
    }
}
