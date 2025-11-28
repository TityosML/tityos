#include "tityos/ty/tensor/Device.h"
#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/Tensor.h"

#include <catch2/catch_all.hpp>
#include <numeric>
#include <vector>

TEST_CASE("Tensor creation", "[Tensor]") {
    REQUIRE_NOTHROW(ty::Tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}));

    REQUIRE_NOTHROW(
        ty::Tensor({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<size_t>({2, 2})));

    REQUIRE_NOTHROW(
        ty::Tensor(std::vector<float>({1.0, 2.0, 3.0, 4.0}), {2, 2}));

    REQUIRE_NOTHROW(ty::Tensor(std::vector<float>({1.0, 2.0, 3.0, 4.0}),
                               std::vector<size_t>({2, 2})));

    REQUIRE_NOTHROW(ty::Tensor(std::vector<float>({1.0, 2.0, 3.0, 4.0}),
                               std::array<size_t, 2>({2, 2})));

    REQUIRE_THROWS(ty::Tensor(std::vector<float>({1.0, 2.0, 3.0, 4.0}),
                              std::array<size_t, 65>({2, 2})));
}

TEST_CASE("Accessing Tensor", "[Tensor]") {
    ty::Tensor example(std::vector<float>({1.0, 2.0, 3.0, 4.0}),
                       std::vector<size_t>({2, 2}));

    CHECK(*static_cast<float*>(example.at({0, 0})) == 1.0f);
    CHECK(*static_cast<float*>(example.at(std::vector<size_t>{0, 1})) == 2.0f);
    CHECK(*static_cast<float*>(example.at(std::array<size_t, 2>{1, 0})) ==
          3.0f);
    CHECK(*static_cast<float*>(example.at({1, 1})) == 4.0f);

    REQUIRE_THROWS(example.at({2, 0}));
    REQUIRE_THROWS(example.at({0, 0, 0}));
}

TEST_CASE("Displaying Tensor", "[Tensor]") {
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}),
                        std::vector<size_t>({2, 2}));

    CHECK(example1.toString() == "[[ 1. 2. ]\n [ 3. 4. ]]");

    ty::Tensor example2(
        std::vector<float>({1.2345678f, 9.8765432f, 10.0f, 1.2f}),
        std::vector<size_t>({2, 2}));

    const std::string expected2 =
        "[[  1.234568  9.876543 ]\n [ 10.000000  1.200000 ]]";
    CHECK(example2.toString() == expected2);

    ty::Tensor example3(std::vector<float>({1.0f, 100.0f, 10.0f, 9999.0f}),
                        std::vector<size_t>({4}));

    const std::string expected3 = "[    1.  100.   10. 9999. ]";
    CHECK(example3.toString() == expected3);

    ty::Tensor example4(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}),
                        std::vector<size_t>({2, 1, 2}));

    const std::string expected4 = "[[[ 1. 2. ]]\n\n [[ 3. 4. ]]]";
    CHECK(example4.toString() == expected4);

    ty::Tensor example5(std::vector<int32_t>({1, 2, 3, 4}),
                        std::vector<size_t>({2, 2, 1}),
                        {ty::DeviceType::CPU, 0}, ty::DType::Int32);

    const std::string expected5 = "[[[ 1 ]\n  [ 2 ]]\n\n [[ 3 ]\n  [ 4 ]]]";
    CHECK(example5.toString() == expected5);
}

TEST_CASE("Tensor copy and move operators", "[Tensor]") {
    ty::Tensor example1({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    SECTION("Copy constructor") {
        auto example2 = example1;

        CHECK(*static_cast<float*>(example2.at({0, 0})) == 1.0);
        CHECK(*static_cast<float*>(example2.at({0, 1})) == 2.0);
        CHECK(*static_cast<float*>(example2.at({1, 0})) == 3.0);
        CHECK(*static_cast<float*>(example2.at({1, 1})) == 4.0);

        *static_cast<float*>(example2.at({0, 0})) = 5.0f;

        CHECK(*static_cast<float*>(example1.at({0, 0})) == 5.0);
        CHECK(*static_cast<float*>(example2.at({0, 0})) == 5.0);
    }

    SECTION("Move constructor") {
        auto example2 = std::move(example1);

        CHECK(*static_cast<float*>(example2.at({0, 0})) == 1.0);
        CHECK(*static_cast<float*>(example2.at({0, 1})) == 2.0);
        CHECK(*static_cast<float*>(example2.at({1, 0})) == 3.0);
        CHECK(*static_cast<float*>(example2.at({1, 1})) == 4.0);

        *static_cast<float*>(example2.at({0, 0})) = 5.0f;

        CHECK(*static_cast<float*>(example2.at({0, 0})) == 5.0);
    }
}