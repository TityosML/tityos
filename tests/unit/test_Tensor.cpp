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

    REQUIRE_NOTHROW(ty::Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2},
                               {ty::DeviceType::CPU, 0}, ty::DType::Float64));

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

    ty::Tensor example6(
        std::vector<int8_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
        std::vector<size_t>({7, 7}), {ty::DeviceType::CPU, 0}, ty::DType::Int8);

    const std::string expected6 =
        "[[ 1 1 1 ... 1 1 1 ]\n [ 1 1 1 ... 1 1 1 ]\n [ 1 1 1 ... 1 1 1 "
        "]\n ...\n "
        "[ 1 1 1 ... 1 1 1 ]\n [ 1 1 1 ... 1 1 1 ]\n [ 1 1 1 ... 1 1 1 ]]";
    CHECK(example6.toString() == expected6);

    std::vector<int32_t> seq_data(9 * 7 * 4);
    std::iota(seq_data.begin(), seq_data.end(), 1);

    ty::Tensor example8(seq_data, std::vector<size_t>({9, 7, 4}),
                        {ty::DeviceType::CPU, 0}, ty::DType::Int32);

    const std::string expected8 = "[[[   1   2 ...   3   4 ]\n"
                                  "  [   5   6 ...   7   8 ]\n"
                                  "  [   9  10 ...  11  12 ]\n"
                                  "  ...\n"
                                  "  [  25  26 ...  27  28 ]]\n\n"
                                  " ...\n\n"
                                  " [[ 225 226 ... 227 228 ]\n"
                                  "  [ 229 230 ... 231 232 ]\n"
                                  "  ...\n"
                                  "  [ 249 250 ... 251 252 ]]]";

    CHECK(example8.toString() == expected8);
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