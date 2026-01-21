#include "tityos/ty/ops/add.h"
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

TEST_CASE("Tensor CUDA Addition", "[Operation][Pointwise]") {

    // Floats
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}),
                        std::vector<size_t>({2, 2}), {ty::DeviceType::CUDA, 0});
    ty::Tensor example2(std::vector<float>({5.0f, 6.0f, 7.0f, 8.0f}),
                        std::vector<size_t>({2, 2}), {ty::DeviceType::CUDA, 0});

    auto result1 = ty::add(example1, example2);

    float a1, a2, a3, a4;
    cudaMemcpy(&a1, result1.at({0, 0}), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&a2, result1.at({0, 1}), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&a3, result1.at({1, 0}), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&a4, result1.at({1, 1}), sizeof(float), cudaMemcpyDeviceToHost);

    CHECK(a1 == 6.0f);
    CHECK(a2 == 8.0f);
    CHECK(a3 == 10.0f);
    CHECK(a4 == 12.0f);

    // Floats with broadcasting
    ty::Tensor example3(
        std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}),
        std::vector<size_t>({3, 2}), {ty::DeviceType::CUDA, 0});

    ty::Tensor example4(std::vector<float>({10.0f, 20.0f}),
                        std::vector<size_t>({2}), {ty::DeviceType::CUDA, 0});

    auto result2 = ty::add(example3, example4);

    float b1, b2, b3, b4, b5, b6;
    cudaMemcpy(&b1, result2.at({0, 0}), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b2, result2.at({0, 1}), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b3, result2.at({1, 0}), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b4, result2.at({1, 1}), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b5, result2.at({2, 0}), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b6, result2.at({2, 1}), sizeof(float), cudaMemcpyDeviceToHost);

    CHECK(b1 == 11.0f);
    CHECK(b2 == 22.0f);

    CHECK(b3 == 13.0f);
    CHECK(b4 == 24.0f);

    CHECK(b5 == 15.0f);
    CHECK(b6 == 26.0f);

    // Int32
    ty::Tensor example5(std::vector<int>({1, 2, 3, 4}),
                        std::vector<size_t>({2, 2}), {ty::DeviceType::CUDA, 0},
                        ty::DType::Int32);
    ty::Tensor example6(std::vector<int>({5, 6, 7, 8}),
                        std::vector<size_t>({2, 2}), {ty::DeviceType::CUDA, 0},
                        ty::DType::Int32);

    auto result3 = ty::add(example5, example6);

    int c1, c2, c3, c4;
    cudaMemcpy(&c1, result3.at({0, 0}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c2, result3.at({0, 1}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c3, result3.at({1, 0}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c4, result3.at({1, 1}), sizeof(int), cudaMemcpyDeviceToHost);

    CHECK(c1 == 6);
    CHECK(c2 == 8);
    CHECK(c3 == 10);
    CHECK(c4 == 12);

    // Int32 with broadcasting
    ty::Tensor example7(std::vector<int>({1, 2, 3, 4, 5, 6}),
                        std::vector<size_t>({1, 3, 2}),
                        {ty::DeviceType::CUDA, 0}, ty::DType::Int32);

    ty::Tensor example8(std::vector<int>({10, 20, 30, 40, 50, 60, 70, 80}),
                        std::vector<size_t>({4, 1, 2}),
                        {ty::DeviceType::CUDA, 0}, ty::DType::Int32);

    auto result4 = ty::add(example7, example8);

    int d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11;
    cudaMemcpy(&d1, result4.at({0, 0, 0}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d2, result4.at({0, 0, 1}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d3, result4.at({0, 1, 0}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d4, result4.at({0, 2, 1}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d5, result4.at({1, 0, 0}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d6, result4.at({1, 1, 1}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d7, result4.at({1, 2, 0}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d8, result4.at({2, 0, 1}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d9, result4.at({2, 1, 0}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d10, result4.at({3, 2, 0}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&d11, result4.at({3, 2, 1}), sizeof(int),
               cudaMemcpyDeviceToHost);

    CHECK(d1 == 11);
    CHECK(d2 == 22);
    CHECK(d3 == 13);
    CHECK(d4 == 26);

    CHECK(d5 == 31);
    CHECK(d6 == 44);
    CHECK(d7 == 35);

    CHECK(d8 == 62);
    CHECK(d9 == 53);

    CHECK(d10 == 75);
    CHECK(d11 == 86);
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
