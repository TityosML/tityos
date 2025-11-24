#include <catch2/catch_all.hpp>

#include "tityos/ty/tensor/Tensor.h"

TEST_CASE("Tensor creation", "[Tensor]") {
    REQUIRE_NOTHROW(
        ty::Tensor(std::vector<float>({1.0, 2.0, 3.0, 4.0}), {2, 2})
    );

    REQUIRE_NOTHROW(
        ty::Tensor(std::vector<float>({1.0, 2.0, 3.0, 4.0}), std::vector<size_t>({2, 2}))
    );

    REQUIRE_NOTHROW(
        ty::Tensor(std::vector<float>({1.0, 2.0, 3.0, 4.0}), std::array<size_t, 2>({2, 2}))
    );
}

TEST_CASE("Accessing Tensor", "[Tensor]") {
    ty::Tensor example(std::vector<float>({1.0, 2.0, 3.0, 4.0}), {2, 2});

    CHECK(*(float*)example.at({0, 0}) == 1.0);
    CHECK(*(float*)example.at(std::vector<size_t>({0, 1})) == 2.0);
    CHECK(*(float*)example.at(std::array<size_t, 2>({1, 0})) == 3.0);
    CHECK(*(float*)example.at({1, 1}) == 4.0);

    REQUIRE_THROWS(example.at({2, 0}));
    REQUIRE_THROWS(example.at({0, 0, 0}));
}