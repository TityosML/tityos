#include <catch2/catch_all.hpp>

#include "tityos/ty/tensor/Tensor.h"

#include <numeric>
#include <vector>

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

TEST_CASE("Displaying Tensor", "[Tensor]") {
    // TODO: update with correct toString functionality
    ty::Tensor example1(std::vector<float>({1.0, 2.0, 3.0, 4.0}), {2, 2});
   
    std::vector<float> v(4*3*2*1);
    std::iota(std::begin(v), std::end(v), 0);

    ty::Tensor example2(v, {4,3,2,1});

    CHECK(example1.toString() == "[[ 1.000000 2.000000 ]\n[ 3.000000 4.000000 ]]");

}