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

#ifdef TITYOS_USE_CUDA
TEST_CASE("TensorStorage can be allocated to GPU with CUDA",
          "[TensorStorage]") {
    REQUIRE_NOTHROW(ty::internal::TensorStorage(4 * sizeof(int),
                                                {ty::DeviceType::CUDA, 0}));
    ty::internal::TensorStorage gpuStorage(4 * sizeof(int),
                                           {ty::DeviceType::CUDA, 0});
    SECTION("Begin pointer can be created") {
        REQUIRE_NOTHROW(gpuStorage.begin());
    }
    SECTION("End pointer can be created") {
        REQUIRE_NOTHROW(gpuStorage.end());
    }
}
#else
TEST_CASE("TensorStorage fails to be allocated to CUDA when CUDA not used",
          "[TensorStorage]") {
    REQUIRE_THROWS(ty::internal::TensorStorage(4 * sizeof(int),
                                               {ty::DeviceType::CUDA, 0}));
}
#endif
