#include "tityos/ty/ops/add.h"
#include "tityos/ty/ops/bmm.h"
#include "tityos/ty/ops/contiguous.h"
#include "tityos/ty/ops/expand.h"
#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <catch2/catch_all.hpp>

TEST_CASE("Tensor Addition", "[Operation][Pointwise]") {

    // Floats
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}),
                        std::vector<size_t>({2, 2}));
    ty::Tensor example2(std::vector<float>({5.0f, 6.0f, 7.0f, 8.0f}),
                        std::vector<size_t>({2, 2}));

    auto result1 = ty::add(example1, example2);

    CHECK(result1.elemAt<float>({0, 0}) == 6.0f);
    CHECK(result1.elemAt<float>({0, 1}) == 8.0f);
    CHECK(result1.elemAt<float>({1, 0}) == 10.0f);
    CHECK(result1.elemAt<float>({1, 1}) == 12.0f);

    // Floats with broadcasting
    ty::Tensor example3(
        std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}),
        std::vector<size_t>({3, 2}));

    ty::Tensor example4(std::vector<float>({10.0f, 20.0f}),
                        std::vector<size_t>({2}));

    auto result2 = ty::add(example3, example4);

    CHECK(result2.elemAt<float>({0, 0}) == 11.0f);
    CHECK(result2.elemAt<float>({0, 1}) == 22.0f);

    CHECK(result2.elemAt<float>({1, 0}) == 13.0f);
    CHECK(result2.elemAt<float>({1, 1}) == 24.0f);

    CHECK(result2.elemAt<float>({2, 0}) == 15.0f);
    CHECK(result2.elemAt<float>({2, 1}) == 26.0f);

    // Int32
    ty::Tensor example5(std::vector<int>({1, 2, 3, 4}),
                        std::vector<size_t>({2, 2}), {ty::DeviceType::CPU, 0},
                        ty::DType::Int32);
    ty::Tensor example6(std::vector<int>({5, 6, 7, 8}),
                        std::vector<size_t>({2, 2}), {ty::DeviceType::CPU, 0},
                        ty::DType::Int32);

    auto result3 = ty::add(example5, example6);

    CHECK(result3.elemAt<int>({0, 0}) == 6);
    CHECK(result3.elemAt<int>({0, 1}) == 8);
    CHECK(result3.elemAt<int>({1, 0}) == 10);
    CHECK(result3.elemAt<int>({1, 1}) == 12);

    // Int32 with broadcasting
    ty::Tensor example7(std::vector<int>({1, 2, 3, 4, 5, 6}),
                        std::vector<size_t>({1, 3, 2}),
                        {ty::DeviceType::CPU, 0}, ty::DType::Int32);

    ty::Tensor example8(std::vector<int>({10, 20, 30, 40, 50, 60, 70, 80}),
                        std::vector<size_t>({4, 1, 2}),
                        {ty::DeviceType::CPU, 0}, ty::DType::Int32);

    auto result4 = ty::add(example7, example8);

    CHECK(result4.elemAt<int>({0, 0, 0}) == 11);
    CHECK(result4.elemAt<int>({0, 0, 1}) == 22);
    CHECK(result4.elemAt<int>({0, 1, 0}) == 13);
    CHECK(result4.elemAt<int>({0, 2, 1}) == 26);

    CHECK(result4.elemAt<int>({1, 0, 0}) == 31);
    CHECK(result4.elemAt<int>({1, 1, 1}) == 44);
    CHECK(result4.elemAt<int>({1, 2, 0}) == 35);

    CHECK(result4.elemAt<int>({2, 0, 1}) == 62);
    CHECK(result4.elemAt<int>({2, 1, 0}) == 53);

    CHECK(result4.elemAt<int>({3, 2, 0}) == 75);
    CHECK(result4.elemAt<int>({3, 2, 1}) == 86);
}

TEST_CASE("Tensor Expand", "[Operation][Pointwise]") {
    // Floats
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}),
                        std::vector<size_t>({2, 2}));

    auto result1 = ty::expand(example1, {4, 2, 2});

    CHECK(result1.elemAt<float>({0, 0, 0}) == 1.0f);
    CHECK(result1.elemAt<float>({0, 0, 1}) == 2.0f);
    CHECK(result1.elemAt<float>({0, 1, 0}) == 3.0f);
    CHECK(result1.elemAt<float>({0, 1, 1}) == 4.0f);

    CHECK(result1.elemAt<float>({1, 0, 0}) == 1.0f);
    CHECK(result1.elemAt<float>({2, 0, 1}) == 2.0f);
    CHECK(result1.elemAt<float>({3, 1, 0}) == 3.0f);
    CHECK(result1.elemAt<float>({3, 1, 1}) == 4.0f);
}

TEST_CASE("Tensor CPU contiguous", "[Operation]") {
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}),
                        std::vector<size_t>({2, 2}));

    ty::Tensor exampleSliced(std::make_shared<ty::internal::BaseTensor>(
        example1.getBaseTensor()->slice(1, 0, 1)));

    auto result = ty::contiguous(exampleSliced);

    CHECK(exampleSliced.isContiguous() == false);
    CHECK(result.isContiguous() == true);

    CHECK(exampleSliced.elemAt<float>({0, 0}) == result.elemAt<float>({0, 0}));
    CHECK(exampleSliced.elemAt<float>({1, 0}) == result.elemAt<float>({1, 0}));

    CHECK(exampleSliced.elemAt<float>({0, 0}) == 1.0f);
}

TEST_CASE("Tensor CPU Batch Matrix-Matrix Multiplication (large with AVX2 "
          "optimization)",
          "[Operation]") {
    // Float
    std::vector<float> data1(64);
    std::vector<float> data2(64, 0.0f);

    for (int i = 0; i < 64; ++i) {
        data1[i] = static_cast<float>(i + 1);

        if (i / 8 == i % 8) {
            data2[i] = 1.0f;
        }
    }

    ty::Tensor example1(data1, {1, 8, 8});
    ty::Tensor example2(data2, {1, 8, 8});

    auto result1 = ty::bmm(example1, example2);

    for (size_t y = 0; y < 8; ++y) {
        for (size_t x = 0; x < 8; ++x) {
            float expected = static_cast<float>(y * 8 + x + 1);
            CHECK(result1.elemAt<float>({0, y, x}) == expected);
        }
    }

    // Int64
    std::vector<int64_t> data3(64);
    std::vector<int64_t> data4(64, 0);

    for (int64_t i = 0; i < 64; ++i) {
        data3[i] = static_cast<int64_t>(i + 1);

        if (i / 8 == i % 8) {
            data4[i] = 1;
        }
    }

    ty::Tensor example3(data3, {1, 8, 8}, {ty::DeviceType::CPU, 0},
                        ty::DType::Int64);
    ty::Tensor example4(data4, {1, 8, 8}, {ty::DeviceType::CPU, 0},
                        ty::DType::Int64);

    auto result2 = ty::bmm(example3, example4);

    for (size_t y = 0; y < 8; ++y) {
        for (size_t x = 0; x < 8; ++x) {
            int64_t expected = static_cast<int64_t>(y * 8 + x + 1);
            CHECK(result2.elemAt<int64_t>({0, y, x}) == expected);
        }
    }

    // Int8
    std::vector<int8_t> data5(64);
    std::vector<int8_t> data6(64, 0);

    for (int8_t i = 0; i < 64; ++i) {
        data5[i] = static_cast<int8_t>(i + 1);

        if (i / 8 == i % 8) {
            data6[i] = 1;
        }
    }

    ty::Tensor example5(data5, {1, 8, 8}, {ty::DeviceType::CPU, 0},
                        ty::DType::Int8);
    ty::Tensor example6(data6, {1, 8, 8}, {ty::DeviceType::CPU, 0},
                        ty::DType::Int8);

    auto result3 = ty::bmm(example5, example6);

    for (size_t y = 0; y < 8; ++y) {
        for (size_t x = 0; x < 8; ++x) {
            int8_t expected = static_cast<int8_t>(y * 8 + x + 1);
            CHECK(result3.elemAt<int8_t>({0, y, x}) == expected);
        }
    }
}

TEST_CASE("Tensor CPU Batch Matrix-Matrix Multiplication", "[Operation]") {
    // Floats 2x2
    ty::Tensor example1(
        std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                            9.0f, 10.0f, 11.0f, 12.0f}),
        std::vector<size_t>({3, 2, 2}));
    ty::Tensor example2(
        std::vector<float>({13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                            20.0f, 21.0f, 22.0f, 23.0f, 24.0f}),
        std::vector<size_t>({3, 2, 2}));

    // note tests are in form ayx
    auto result1 = ty::bmm(example1, example2);

    // z y x
    CHECK(result1.elemAt<float>({0, 0, 0}) == 43.0f);
    CHECK(result1.elemAt<float>({0, 0, 1}) == 46.0f);
    CHECK(result1.elemAt<float>({0, 1, 0}) == 99.0f);
    CHECK(result1.elemAt<float>({0, 1, 1}) == 106.0f);

    CHECK(result1.elemAt<float>({1, 0, 0}) == 199.0f);
    CHECK(result1.elemAt<float>({1, 0, 1}) == 210.0f);
    CHECK(result1.elemAt<float>({1, 1, 0}) == 271.0f);
    CHECK(result1.elemAt<float>({1, 1, 1}) == 286.0f);

    CHECK(result1.elemAt<float>({2, 0, 0}) == 419.0f);
    CHECK(result1.elemAt<float>({2, 0, 1}) == 438.0f);
    CHECK(result1.elemAt<float>({2, 1, 0}) == 507.0f);
    CHECK(result1.elemAt<float>({2, 1, 1}) == 530.0f);

    // Floats 3x2 @ 2x3 -> 3x3
    ty::Tensor example3(
        std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                            9.0f, 10.0f, 11.0f, 12.0f}),
        std::vector<size_t>({2, 3, 2}));
    ty::Tensor example4(
        std::vector<float>({13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                            20.0f, 21.0f, 22.0f, 23.0f, 24.0f}),
        std::vector<size_t>({2, 2, 3}));

    auto result2 = ty::bmm(example3, example4);

    // d-block (z = 0)
    CHECK(result2.elemAt<float>({0, 0, 0}) == 45.0f);
    CHECK(result2.elemAt<float>({0, 0, 1}) == 48.0f);
    CHECK(result2.elemAt<float>({0, 0, 2}) == 51.0f);

    CHECK(result2.elemAt<float>({0, 1, 0}) == 103.0f);
    CHECK(result2.elemAt<float>({0, 1, 1}) == 110.0f);
    CHECK(result2.elemAt<float>({0, 1, 2}) == 117.0f);

    CHECK(result2.elemAt<float>({0, 2, 0}) == 161.0f);
    CHECK(result2.elemAt<float>({0, 2, 1}) == 172.0f);
    CHECK(result2.elemAt<float>({0, 2, 2}) == 183.0f);

    // e-block (z = 1)
    CHECK(result2.elemAt<float>({1, 0, 0}) == 309.0f);
    CHECK(result2.elemAt<float>({1, 0, 1}) == 324.0f);
    CHECK(result2.elemAt<float>({1, 0, 2}) == 339.0f);

    CHECK(result2.elemAt<float>({1, 1, 0}) == 391.0f);
    CHECK(result2.elemAt<float>({1, 1, 1}) == 410.0f);
    CHECK(result2.elemAt<float>({1, 1, 2}) == 429.0f);

    CHECK(result2.elemAt<float>({1, 2, 0}) == 473.0f);
    CHECK(result2.elemAt<float>({1, 2, 1}) == 496.0f);
    CHECK(result2.elemAt<float>({1, 2, 2}) == 519.0f);

    // Int32 2x2
    ty::Tensor example5(
        std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
        std::vector<size_t>({3, 2, 2}), {ty::DeviceType::CPU, 0},
        ty::DType::Int32);
    ty::Tensor example6(
        std::vector<int>({13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
        std::vector<size_t>({3, 2, 2}), {ty::DeviceType::CPU, 0},
        ty::DType::Int32);

    auto result3 = ty::bmm(example5, example6);

    CHECK(result3.elemAt<int>({0, 0, 0}) == 43);
    CHECK(result3.elemAt<int>({0, 0, 1}) == 46);
    CHECK(result3.elemAt<int>({0, 1, 0}) == 99);
    CHECK(result3.elemAt<int>({0, 1, 1}) == 106);

    CHECK(result3.elemAt<int>({1, 0, 0}) == 199);
    CHECK(result3.elemAt<int>({1, 0, 1}) == 210);
    CHECK(result3.elemAt<int>({1, 1, 0}) == 271);
    CHECK(result3.elemAt<int>({1, 1, 1}) == 286);

    CHECK(result3.elemAt<int>({2, 0, 0}) == 419);
    CHECK(result3.elemAt<int>({2, 0, 1}) == 438);
    CHECK(result3.elemAt<int>({2, 1, 0}) == 507);
    CHECK(result3.elemAt<int>({2, 1, 1}) == 530);

    // Ints 3x2 @ 2x3 -> 3x3
    ty::Tensor example7(
        std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
        std::vector<size_t>({2, 3, 2}), {ty::DeviceType::CPU, 0},
        ty::DType::Int32);
    ty::Tensor example8(
        std::vector<int>({13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
        std::vector<size_t>({2, 2, 3}), {ty::DeviceType::CPU, 0},
        ty::DType::Int32);

    auto result4 = ty::bmm(example7, example8);

    // z y x
    CHECK(result4.elemAt<int>({0, 0, 0}) == 45);
    CHECK(result4.elemAt<int>({0, 0, 1}) == 48);
    CHECK(result4.elemAt<int>({0, 0, 2}) == 51);

    CHECK(result4.elemAt<int>({0, 1, 0}) == 103);
    CHECK(result4.elemAt<int>({0, 1, 1}) == 110);
    CHECK(result4.elemAt<int>({0, 1, 2}) == 117);

    CHECK(result4.elemAt<int>({0, 2, 0}) == 161);
    CHECK(result4.elemAt<int>({0, 2, 1}) == 172);
    CHECK(result4.elemAt<int>({0, 2, 2}) == 183);

    CHECK(result4.elemAt<int>({1, 0, 0}) == 309);
    CHECK(result4.elemAt<int>({1, 0, 1}) == 324);
    CHECK(result4.elemAt<int>({1, 0, 2}) == 339);

    CHECK(result4.elemAt<int>({1, 1, 0}) == 391);
    CHECK(result4.elemAt<int>({1, 1, 1}) == 410);
    CHECK(result4.elemAt<int>({1, 1, 2}) == 429);

    CHECK(result4.elemAt<int>({1, 2, 0}) == 473);
    CHECK(result4.elemAt<int>({1, 2, 1}) == 496);
    CHECK(result4.elemAt<int>({1, 2, 2}) == 519);
}