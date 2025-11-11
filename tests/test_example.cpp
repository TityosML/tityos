#include <catch2/catch_all.hpp>

#include "tityos/ty/tensor/ByteArray.h"

TEST_CASE("Example ctest", "[example]") {
    REQUIRE_NOTHROW([&]() { ByteArray exampleArr(4 * sizeof(int)); });
}