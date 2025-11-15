#include <catch2/catch_all.hpp>

#include "tityos/ty/tensor/ByteArray.cpp"

TEST_CASE("ByteArrays can be allocated", "[ByteArray]") {
    REQUIRE_NOTHROW([&]() { ty::internal::ByteArray exampleArr(4 * sizeof(int)); });
}