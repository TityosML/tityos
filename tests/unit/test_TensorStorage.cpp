#include <catch2/catch_all.hpp>

#include "tityos/ty/tensor/TensorStorage.h"

TEST_CASE("TensorStorage can be allocated to CPU", "[TensorStorage]") {
    REQUIRE_NOTHROW(ty::internal::TensorStorage(4 * sizeof(int)));
}

#ifdef TITYOS_USE_CUDA
    TEST_CASE("TensorStorage can be allocated to GPU with CUDA", "[TensorStorage]") {
        REQUIRE_NOTHROW(ty::internal::TensorStorage(4 * sizeof(int), {ty::DeviceType::CUDA, 0}));
    }
#else
    TEST_CASE("TensorStorage fails to be allocated to CUDA when CUDA not used", "[TensorStorage]") {
        REQUIRE_THROWS(ty::internal::TensorStorage(4 * sizeof(int), {ty::DeviceType::CUDA, 0}));
    }
#endif
