#include "tityos/ty/ops/add.h"
#include "tityos/ty/ops/bmm.h"
#include "tityos/ty/ops/expand.h"
#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/ShapeStrides.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Benchmark Tensor Add Broadcasting CPU (large tensors)",
          "[benchmark][broadcast][add]") {
    constexpr size_t D0 = 1;
    constexpr size_t D1 = 64;
    constexpr size_t D2 = 128;

    constexpr size_t E0 = 64;
    constexpr size_t E1 = 1;
    constexpr size_t E2 = 128;

    std::vector<int> dataA(D0 * D1 * D2);
    for (size_t i = 0; i < dataA.size(); ++i) {
        dataA[i] = static_cast<int>(i % 17);
    }

    ty::Tensor A(dataA, std::vector<size_t>({D0, D1, D2}),
                 {ty::DeviceType::CPU, 0}, ty::DType::Int32);

    std::vector<int> dataB(E0 * E1 * E2);
    for (size_t i = 0; i < dataB.size(); i++) {
        dataB[i] = static_cast<int>((i % 31) + 1);
    }

    ty::Tensor B(dataB, std::vector<size_t>({E0, E1, E2}),
                 {ty::DeviceType::CPU, 0}, ty::DType::Int32);

    BENCHMARK("add broadcast") {
        auto result = ty::add(A, B);
        return result;
    };
}

TEST_CASE("Benchmark Tensor Add Avx2(large tensors)",
          "[benchmark][add]") {
    constexpr size_t D0 = 64;
    constexpr size_t D1 = 64;
    constexpr size_t D2 = 128;

    std::vector<int> dataA(D0 * D1 * D2);
    for (size_t i = 0; i < dataA.size(); ++i) {
        dataA[i] = static_cast<int>(i % 17);
    }

    ty::Tensor A(dataA, std::vector<size_t>({D0, D1, D2}),
                 {ty::DeviceType::CPU, 0}, ty::DType::Int32);

    std::vector<int> dataB(D0 * D1 * D2);
    for (size_t i = 0; i < dataB.size(); i++) {
        dataB[i] = static_cast<int>((i % 31) + 1);
    }

    ty::Tensor B(dataB, std::vector<size_t>({D0, D1, D2}),
                 {ty::DeviceType::CPU, 0}, ty::DType::Int32);

    BENCHMARK("add contiguous") {
        auto result = ty::add(A, B);
        return result;
    };
}

TEST_CASE("Benchmark Batched MatMul CPU (large tensors)",
          "[benchmark][bmm][matmul]") {
    constexpr size_t BATCH = 32;
    constexpr size_t M = 64;
    constexpr size_t N = 128;
    constexpr size_t K = 64;

    std::vector<int> dataA(BATCH * M * N);
    for (size_t i = 0; i < dataA.size(); ++i) {
        dataA[i] = static_cast<int>(i % 13);
    }

    ty::Tensor A(dataA, std::vector<size_t>{BATCH, M, N}, {ty::DeviceType::CPU, 0},
                 ty::DType::Int32);

    std::vector<int> dataB(BATCH * N * K);
    for (size_t i = 0; i < dataB.size(); ++i) {
        dataB[i] = static_cast<int>((i % 7) + 1);
    }

    ty::Tensor B(dataB, std::vector<size_t>{BATCH, N, K},
                        {ty::DeviceType::CPU, 0}, ty::DType::Int32);

    BENCHMARK("bmm int32 CPU") {
        auto result = ty::bmm(A, B);
        return result;
    };
}