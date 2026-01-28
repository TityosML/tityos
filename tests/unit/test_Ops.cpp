#include "tityos/ty/ops/add.h"
#include "tityos/ty/ops/bmm.h"
#include "tityos/ty/ops/contiguous.h"
#include "tityos/ty/ops/expand.h"
#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <catch2/catch_all.hpp>

#ifdef TITYOS_BUILD_CUDA
    #include <cuda_runtime.h>
#endif

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

#ifdef TITYOS_BUILD_CUDA
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
#endif

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

#ifdef TITYOS_BUILD_CUDA
TEST_CASE("Tensor CUDA Batch Matrix-Matrix Multiplication", "[Operation]") {

    // Floats 2x2
    ty::Tensor example1(
        std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                            9.0f, 10.0f, 11.0f, 12.0f}),
        std::vector<size_t>({3, 2, 2}), {ty::DeviceType::CUDA, 0});
    ty::Tensor example2(
        std::vector<float>({13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                            20.0f, 21.0f, 22.0f, 23.0f, 24.0f}),
        std::vector<size_t>({3, 2, 2}), {ty::DeviceType::CUDA, 0});

    // note tests are in form ayx
    auto result1 = ty::bmm(example1, example2);
    //                           z  y  x
    float a00, a01, a10, a11;
    cudaMemcpy(&a00, result1.at({0, 0, 0}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&a01, result1.at({0, 0, 1}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&a10, result1.at({0, 1, 0}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&a11, result1.at({0, 1, 1}), sizeof(float),
               cudaMemcpyDeviceToHost);

    float b00, b01, b10, b11;
    cudaMemcpy(&b00, result1.at({1, 0, 0}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&b01, result1.at({1, 0, 1}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&b10, result1.at({1, 1, 0}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&b11, result1.at({1, 1, 1}), sizeof(float),
               cudaMemcpyDeviceToHost);

    float c00, c01, c10, c11;
    cudaMemcpy(&c00, result1.at({2, 0, 0}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&c01, result1.at({2, 0, 1}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&c10, result1.at({2, 1, 0}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&c11, result1.at({2, 1, 1}), sizeof(float),
               cudaMemcpyDeviceToHost);

    CHECK(a00 == 43.0f);
    CHECK(a01 == 46.0f);
    CHECK(a10 == 99.0f);
    CHECK(a11 == 106.0f);

    CHECK(b00 == 199.0f);
    CHECK(b01 == 210.0f);
    CHECK(b10 == 271.0f);
    CHECK(b11 == 286.0f);

    CHECK(c00 == 419.0f);
    CHECK(c01 == 438.0f);
    CHECK(c10 == 507.0f);
    CHECK(c11 == 530.0f);

    // Floats 3x2 @ 2x3 -> 3x3
    ty::Tensor example3(
        std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                            9.0f, 10.0f, 11.0f, 12.0f}),
        std::vector<size_t>({2, 3, 2}), {ty::DeviceType::CUDA, 0});
    ty::Tensor example4(
        std::vector<float>({13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                            20.0f, 21.0f, 22.0f, 23.0f, 24.0f}),
        std::vector<size_t>({2, 2, 3}), {ty::DeviceType::CUDA, 0});

    auto result2 = ty::bmm(example3, example4);

    float d00, d01, d02, d10, d11, d12, d20, d21, d22;
    cudaMemcpy(&d00, result2.at({0, 0, 0}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&d01, result2.at({0, 0, 1}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&d02, result2.at({0, 0, 2}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&d10, result2.at({0, 1, 0}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&d11, result2.at({0, 1, 1}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&d12, result2.at({0, 1, 2}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&d20, result2.at({0, 2, 0}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&d21, result2.at({0, 2, 1}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&d22, result2.at({0, 2, 2}), sizeof(float),
               cudaMemcpyDeviceToHost);

    float e00, e01, e02, e10, e11, e12, e20, e21, e22;
    cudaMemcpy(&e00, result2.at({1, 0, 0}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&e01, result2.at({1, 0, 1}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&e02, result2.at({1, 0, 2}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&e10, result2.at({1, 1, 0}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&e11, result2.at({1, 1, 1}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&e12, result2.at({1, 1, 2}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&e20, result2.at({1, 2, 0}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&e21, result2.at({1, 2, 1}), sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&e22, result2.at({1, 2, 2}), sizeof(float),
               cudaMemcpyDeviceToHost);

    CHECK(d00 == 45.0f);
    CHECK(d01 == 48.0f);
    CHECK(d02 == 51.0f);
    CHECK(d10 == 103.0f);
    CHECK(d11 == 110.0f);
    CHECK(d12 == 117.0f);
    CHECK(d20 == 161.0f);
    CHECK(d21 == 172.0f);
    CHECK(d22 == 183.0f);

    CHECK(e00 == 309.0f);
    CHECK(e01 == 324.0f);
    CHECK(e02 == 339.0f);
    CHECK(e10 == 391.0f);
    CHECK(e11 == 410.0f);
    CHECK(e12 == 429.0f);
    CHECK(e20 == 473.0f);
    CHECK(e21 == 496.0f);
    CHECK(e22 == 519.0f);

    // Int32 2x2
    ty::Tensor example5(
        std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
        std::vector<size_t>({3, 2, 2}), {ty::DeviceType::CUDA, 0},
        ty::DType::Int32);
    ty::Tensor example6(
        std::vector<int>({13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
        std::vector<size_t>({3, 2, 2}), {ty::DeviceType::CUDA, 0},
        ty::DType::Int32);

    auto result3 = ty::bmm(example5, example6);

    int ai00, ai01, ai10, ai11;
    cudaMemcpy(&ai00, result3.at({0, 0, 0}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&ai01, result3.at({0, 0, 1}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&ai10, result3.at({0, 1, 0}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&ai11, result3.at({0, 1, 1}), sizeof(int),
               cudaMemcpyDeviceToHost);

    int bi00, bi01, bi10, bi11;
    cudaMemcpy(&bi00, result3.at({1, 0, 0}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&bi01, result3.at({1, 0, 1}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&bi10, result3.at({1, 1, 0}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&bi11, result3.at({1, 1, 1}), sizeof(int),
               cudaMemcpyDeviceToHost);

    int ci00, ci01, ci10, ci11;
    cudaMemcpy(&ci00, result3.at({2, 0, 0}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&ci01, result3.at({2, 0, 1}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&ci10, result3.at({2, 1, 0}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&ci11, result3.at({2, 1, 1}), sizeof(int),
               cudaMemcpyDeviceToHost);

    CHECK(ai00 == 43);
    CHECK(ai01 == 46);
    CHECK(ai10 == 99);
    CHECK(ai11 == 106);

    CHECK(bi00 == 199);
    CHECK(bi01 == 210);
    CHECK(bi10 == 271);
    CHECK(bi11 == 286);

    CHECK(ci00 == 419);
    CHECK(ci01 == 438);
    CHECK(ci10 == 507);
    CHECK(ci11 == 530);

    // Ints 3x2 @ 2x3 -> 3x3
    ty::Tensor example7(
        std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
        std::vector<size_t>({2, 3, 2}), {ty::DeviceType::CUDA, 0},
        ty::DType::Int32);
    ty::Tensor example8(
        std::vector<int>({13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
        std::vector<size_t>({2, 2, 3}), {ty::DeviceType::CUDA, 0},
        ty::DType::Int32);

    auto result4 = ty::bmm(example7, example8);

    int di00, di01, di02, di10, di11, di12, di20, di21, di22;
    cudaMemcpy(&di00, result4.at({0, 0, 0}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&di01, result4.at({0, 0, 1}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&di02, result4.at({0, 0, 2}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&di10, result4.at({0, 1, 0}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&di11, result4.at({0, 1, 1}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&di12, result4.at({0, 1, 2}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&di20, result4.at({0, 2, 0}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&di21, result4.at({0, 2, 1}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&di22, result4.at({0, 2, 2}), sizeof(int),
               cudaMemcpyDeviceToHost);

    int ei00, ei01, ei02, ei10, ei11, ei12, ei20, ei21, ei22;
    cudaMemcpy(&ei00, result4.at({1, 0, 0}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&ei01, result4.at({1, 0, 1}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&ei02, result4.at({1, 0, 2}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&ei10, result4.at({1, 1, 0}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&ei11, result4.at({1, 1, 1}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&ei12, result4.at({1, 1, 2}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&ei20, result4.at({1, 2, 0}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&ei21, result4.at({1, 2, 1}), sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&ei22, result4.at({1, 2, 2}), sizeof(int),
               cudaMemcpyDeviceToHost);

    CHECK(di00 == 45);
    CHECK(di01 == 48);
    CHECK(di02 == 51);
    CHECK(di10 == 103);
    CHECK(di11 == 110);
    CHECK(di12 == 117);
    CHECK(di20 == 161);
    CHECK(di21 == 172);
    CHECK(di22 == 183);

    CHECK(ei00 == 309);
    CHECK(ei01 == 324);
    CHECK(ei02 == 339);
    CHECK(ei10 == 391);
    CHECK(ei11 == 410);
    CHECK(ei12 == 429);
    CHECK(ei20 == 473);
    CHECK(ei21 == 496);
    CHECK(ei22 == 519);
}
#endif

#ifdef TITYOS_BUILD_CUDA
TEST_CASE("Tensor CUDA contiguous", "[Operation]") {
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}),
                        std::vector<size_t>({2, 2}), {ty::DeviceType::CUDA, 0});

    ty::Tensor exampleSliced(std::make_shared<ty::internal::BaseTensor>(
        example1.getBaseTensor()->slice(1, 0, 1)));

    auto result = ty::contiguous(exampleSliced);

    CHECK(exampleSliced.isContiguous() == false);
    CHECK(result.isContiguous() == true);

    float e1, e2;
    cudaMemcpy(&e1, example1.at({0, 0}), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&e2, example1.at({1, 0}), sizeof(float), cudaMemcpyDeviceToHost);

    float r1, r2;
    cudaMemcpy(&r1, result.at({0, 0}), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&r2, result.at({1, 0}), sizeof(float), cudaMemcpyDeviceToHost);

    CHECK(e1 == r1);
    CHECK(e2 == r2);

    CHECK(e1 == 1.0f);
}
#endif