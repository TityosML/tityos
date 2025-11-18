#include <catch2/catch_all.hpp>

#include "tityos/ty/tensor/TensorStorage.h"

TEST_CASE("TensorStorage can be allocated to CPU", "[TensorStorage]") {
    REQUIRE_NOTHROW(ty::internal::TensorStorage(4 * sizeof(int)));
}

TEST_CASE("TensorStorage can be allocated to GPU", "[TensorStorage]") {
    #ifdef TITYOS_USE_CUDA
        REQUIRE_NOTHROW(ty::internal::TensorStorage(4 * sizeof(int), {ty::DeviceType::CUDA, 0}));
    #else
        REQUIRE_THROWS(ty::internal::TensorStorage(4 * sizeof(int), {ty::DeviceType::CUDA, 0}));
    #endif
}