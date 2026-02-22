#include "tityos/ty/ops/add.h"
#include "tityos/ty/ops/bmm.h"
#include "tityos/ty/ops/contiguous.h"
#include "tityos/ty/ops/expand.h"
#include "tityos/ty/ops/toCpu.h"
#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>

TEST_CASE("Tensor CUDA toCpu", "[Operation]") {
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}), std::vector<size_t>({2, 2}),
                        {ty::DeviceType::CUDA, 0});

    auto example1Cpu = ty::toCpu(example1);

    CHECK(example1Cpu.elemAt<float>({0, 0}) == 1.0f);
    CHECK(example1Cpu.elemAt<float>({0, 1}) == 2.0f);
    CHECK(example1Cpu.elemAt<float>({1, 0}) == 3.0f);
    CHECK(example1Cpu.elemAt<float>({1, 1}) == 4.0f);
}

TEST_CASE("Tensor CUDA Addition", "[Operation][Pointwise]") {

    // Floats
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}), std::vector<size_t>({2, 2}),
                        {ty::DeviceType::CUDA, 0});
    ty::Tensor example2(std::vector<float>({5.0f, 6.0f, 7.0f, 8.0f}), std::vector<size_t>({2, 2}),
                        {ty::DeviceType::CUDA, 0});

    auto result1 = ty::add(example1, example2);
    auto result1Cpu = ty::toCpu(result1);

    CHECK(result1Cpu.elemAt<float>({0, 0}) == 6.0f);
    CHECK(result1Cpu.elemAt<float>({0, 1}) == 8.0f);
    CHECK(result1Cpu.elemAt<float>({1, 0}) == 10.0f);
    CHECK(result1Cpu.elemAt<float>({1, 1}) == 12.0f);

    // Floats with broadcasting
    ty::Tensor example3(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}), std::vector<size_t>({3, 2}),
                        {ty::DeviceType::CUDA, 0});

    ty::Tensor example4(std::vector<float>({10.0f, 20.0f}), std::vector<size_t>({2}), {ty::DeviceType::CUDA, 0});

    auto result2 = ty::add(example3, example4);
    auto result2Cpu = ty::toCpu(result2);

    CHECK(result2Cpu.elemAt<float>({0, 0}) == 11.0f);
    CHECK(result2Cpu.elemAt<float>({0, 1}) == 22.0f);

    CHECK(result2Cpu.elemAt<float>({1, 0}) == 13.0f);
    CHECK(result2Cpu.elemAt<float>({1, 1}) == 24.0f);

    CHECK(result2Cpu.elemAt<float>({2, 0}) == 15.0f);
    CHECK(result2Cpu.elemAt<float>({2, 1}) == 26.0f);

    // Int32
    ty::Tensor example5(std::vector<int>({1, 2, 3, 4}), std::vector<size_t>({2, 2}), {ty::DeviceType::CUDA, 0},
                        ty::DType::Int32);
    ty::Tensor example6(std::vector<int>({5, 6, 7, 8}), std::vector<size_t>({2, 2}), {ty::DeviceType::CUDA, 0},
                        ty::DType::Int32);

    auto result3 = ty::add(example5, example6);
    auto result3Cpu = ty::toCpu(result3);

    CHECK(result3Cpu.elemAt<int>({0, 0}) == 6);
    CHECK(result3Cpu.elemAt<int>({0, 1}) == 8);
    CHECK(result3Cpu.elemAt<int>({1, 0}) == 10);
    CHECK(result3Cpu.elemAt<int>({1, 1}) == 12);

    // Int32 with broadcasting
    ty::Tensor example7(std::vector<int>({1, 2, 3, 4, 5, 6}), std::vector<size_t>({1, 3, 2}), {ty::DeviceType::CUDA, 0},
                        ty::DType::Int32);

    ty::Tensor example8(std::vector<int>({10, 20, 30, 40, 50, 60, 70, 80}), std::vector<size_t>({4, 1, 2}),
                        {ty::DeviceType::CUDA, 0}, ty::DType::Int32);

    auto result4 = ty::add(example7, example8);
    auto result4Cpu = ty::toCpu(result4);

    CHECK(result4Cpu.elemAt<int>({0, 0, 0}) == 11);
    CHECK(result4Cpu.elemAt<int>({0, 0, 1}) == 22);
    CHECK(result4Cpu.elemAt<int>({0, 1, 0}) == 13);
    CHECK(result4Cpu.elemAt<int>({0, 2, 1}) == 26);

    CHECK(result4Cpu.elemAt<int>({1, 0, 0}) == 31);
    CHECK(result4Cpu.elemAt<int>({1, 1, 1}) == 44);
    CHECK(result4Cpu.elemAt<int>({1, 2, 0}) == 35);

    CHECK(result4Cpu.elemAt<int>({2, 0, 1}) == 62);
    CHECK(result4Cpu.elemAt<int>({2, 1, 0}) == 53);

    CHECK(result4Cpu.elemAt<int>({3, 2, 0}) == 75);
    CHECK(result4Cpu.elemAt<int>({3, 2, 1}) == 86);
}

TEST_CASE("Tensor CUDA Batch Matrix-Matrix Multiplication", "[Operation]") {

    // Floats 2x2
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}),
                        std::vector<size_t>({3, 2, 2}), {ty::DeviceType::CUDA, 0});
    ty::Tensor example2(
        std::vector<float>({13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f}),
        std::vector<size_t>({3, 2, 2}), {ty::DeviceType::CUDA, 0});

    // note tests are in form ayx
    auto result1 = ty::bmm(example1, example2);
    auto result1Cpu = ty::toCpu(result1);

    // z y x
    CHECK(result1Cpu.elemAt<float>({0, 0, 0}) == 43.0f);
    CHECK(result1Cpu.elemAt<float>({0, 0, 1}) == 46.0f);
    CHECK(result1Cpu.elemAt<float>({0, 1, 0}) == 99.0f);
    CHECK(result1Cpu.elemAt<float>({0, 1, 1}) == 106.0f);

    CHECK(result1Cpu.elemAt<float>({1, 0, 0}) == 199.0f);
    CHECK(result1Cpu.elemAt<float>({1, 0, 1}) == 210.0f);
    CHECK(result1Cpu.elemAt<float>({1, 1, 0}) == 271.0f);
    CHECK(result1Cpu.elemAt<float>({1, 1, 1}) == 286.0f);

    CHECK(result1Cpu.elemAt<float>({2, 0, 0}) == 419.0f);
    CHECK(result1Cpu.elemAt<float>({2, 0, 1}) == 438.0f);
    CHECK(result1Cpu.elemAt<float>({2, 1, 0}) == 507.0f);
    CHECK(result1Cpu.elemAt<float>({2, 1, 1}) == 530.0f);

    // Floats 3x2 @ 2x3 -> 3x3
    ty::Tensor example3(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}),
                        std::vector<size_t>({2, 3, 2}), {ty::DeviceType::CUDA, 0});
    ty::Tensor example4(
        std::vector<float>({13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f}),
        std::vector<size_t>({2, 2, 3}), {ty::DeviceType::CUDA, 0});

    auto result2 = ty::bmm(example3, example4);
    auto result2Cpu = ty::toCpu(result2);

    // d-block (z = 0)
    CHECK(result2Cpu.elemAt<float>({0, 0, 0}) == 45.0f);
    CHECK(result2Cpu.elemAt<float>({0, 0, 1}) == 48.0f);
    CHECK(result2Cpu.elemAt<float>({0, 0, 2}) == 51.0f);

    CHECK(result2Cpu.elemAt<float>({0, 1, 0}) == 103.0f);
    CHECK(result2Cpu.elemAt<float>({0, 1, 1}) == 110.0f);
    CHECK(result2Cpu.elemAt<float>({0, 1, 2}) == 117.0f);

    CHECK(result2Cpu.elemAt<float>({0, 2, 0}) == 161.0f);
    CHECK(result2Cpu.elemAt<float>({0, 2, 1}) == 172.0f);
    CHECK(result2Cpu.elemAt<float>({0, 2, 2}) == 183.0f);

    // e-block (z = 1)
    CHECK(result2Cpu.elemAt<float>({1, 0, 0}) == 309.0f);
    CHECK(result2Cpu.elemAt<float>({1, 0, 1}) == 324.0f);
    CHECK(result2Cpu.elemAt<float>({1, 0, 2}) == 339.0f);

    CHECK(result2Cpu.elemAt<float>({1, 1, 0}) == 391.0f);
    CHECK(result2Cpu.elemAt<float>({1, 1, 1}) == 410.0f);
    CHECK(result2Cpu.elemAt<float>({1, 1, 2}) == 429.0f);

    CHECK(result2Cpu.elemAt<float>({1, 2, 0}) == 473.0f);
    CHECK(result2Cpu.elemAt<float>({1, 2, 1}) == 496.0f);
    CHECK(result2Cpu.elemAt<float>({1, 2, 2}) == 519.0f);

    // Int32 2x2
    ty::Tensor example5(std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), std::vector<size_t>({3, 2, 2}),
                        {ty::DeviceType::CUDA, 0}, ty::DType::Int32);
    ty::Tensor example6(std::vector<int>({13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
                        std::vector<size_t>({3, 2, 2}), {ty::DeviceType::CUDA, 0}, ty::DType::Int32);

    auto result3 = ty::bmm(example5, example6);

    int ai00, ai01, ai10, ai11;
    cudaMemcpy(&ai00, result3.at({0, 0, 0}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ai01, result3.at({0, 0, 1}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ai10, result3.at({0, 1, 0}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ai11, result3.at({0, 1, 1}), sizeof(int), cudaMemcpyDeviceToHost);

    int bi00, bi01, bi10, bi11;
    cudaMemcpy(&bi00, result3.at({1, 0, 0}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&bi01, result3.at({1, 0, 1}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&bi10, result3.at({1, 1, 0}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&bi11, result3.at({1, 1, 1}), sizeof(int), cudaMemcpyDeviceToHost);

    int ci00, ci01, ci10, ci11;
    cudaMemcpy(&ci00, result3.at({2, 0, 0}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ci01, result3.at({2, 0, 1}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ci10, result3.at({2, 1, 0}), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ci11, result3.at({2, 1, 1}), sizeof(int), cudaMemcpyDeviceToHost);

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
    ty::Tensor example7(std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), std::vector<size_t>({2, 3, 2}),
                        {ty::DeviceType::CUDA, 0}, ty::DType::Int32);
    ty::Tensor example8(std::vector<int>({13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
                        std::vector<size_t>({2, 2, 3}), {ty::DeviceType::CUDA, 0}, ty::DType::Int32);

    auto result4 = ty::bmm(example7, example8);
    auto result4Cpu = ty::toCpu(result4);

    // z y x
    CHECK(result4Cpu.elemAt<int>({0, 0, 0}) == 45);
    CHECK(result4Cpu.elemAt<int>({0, 0, 1}) == 48);
    CHECK(result4Cpu.elemAt<int>({0, 0, 2}) == 51);

    CHECK(result4Cpu.elemAt<int>({0, 1, 0}) == 103);
    CHECK(result4Cpu.elemAt<int>({0, 1, 1}) == 110);
    CHECK(result4Cpu.elemAt<int>({0, 1, 2}) == 117);

    CHECK(result4Cpu.elemAt<int>({0, 2, 0}) == 161);
    CHECK(result4Cpu.elemAt<int>({0, 2, 1}) == 172);
    CHECK(result4Cpu.elemAt<int>({0, 2, 2}) == 183);

    CHECK(result4Cpu.elemAt<int>({1, 0, 0}) == 309);
    CHECK(result4Cpu.elemAt<int>({1, 0, 1}) == 324);
    CHECK(result4Cpu.elemAt<int>({1, 0, 2}) == 339);

    CHECK(result4Cpu.elemAt<int>({1, 1, 0}) == 391);
    CHECK(result4Cpu.elemAt<int>({1, 1, 1}) == 410);
    CHECK(result4Cpu.elemAt<int>({1, 1, 2}) == 429);

    CHECK(result4Cpu.elemAt<int>({1, 2, 0}) == 473);
    CHECK(result4Cpu.elemAt<int>({1, 2, 1}) == 496);
    CHECK(result4Cpu.elemAt<int>({1, 2, 2}) == 519);
}

TEST_CASE("Tensor CUDA contiguous", "[Operation]") {
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}), std::vector<size_t>({2, 2}),
                        {ty::DeviceType::CUDA, 0});

    ty::Tensor exampleSliced(std::make_shared<ty::internal::BaseTensor>(example1.getBaseTensor()->slice(1, 0, 1)));

    auto result = ty::contiguous(exampleSliced);

    auto resultCpu = ty::toCpu(result);
    auto exampleSlicedCpu = ty::toCpu(exampleSliced);

    CHECK(exampleSliced.isContiguous() == false);
    CHECK(result.isContiguous() == true);

    CHECK(exampleSlicedCpu.elemAt<float>({0, 0}) == resultCpu.elemAt<float>({0, 0}));
    CHECK(exampleSlicedCpu.elemAt<float>({1, 0}) == resultCpu.elemAt<float>({1, 0}));

    CHECK(exampleSlicedCpu.elemAt<float>({0, 0}) == 1.0f);
}