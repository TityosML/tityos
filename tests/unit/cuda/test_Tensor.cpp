#include "tityos/ty/tensor/Dtype.h"
#include "tityos/ty/tensor/Tensor.h"

#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>

TEST_CASE("Displaying Tensor CUDA", "[Tensor]") {
    ty::Tensor example1(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}), std::vector<size_t>({2, 2}),
                        {ty::DeviceType::CUDA, 0});

    CHECK(example1.toString() == "[[ 1. 2. ]\n [ 3. 4. ]]");

    ty::Tensor example2(std::vector<float>({1.2345678f, 9.8765432f, 10.0f, 1.2f}), std::vector<size_t>({2, 2}),
                        {ty::DeviceType::CUDA, 0});

    const std::string expected2 = "[[  1.234568  9.876543 ]\n [ 10.000000  1.200000 ]]";
    CHECK(example2.toString() == expected2);

    ty::Tensor example3(std::vector<float>({1.0f, 100.0f, 10.0f, 9999.0f}), std::vector<size_t>({4}),
                        {ty::DeviceType::CUDA, 0});

    const std::string expected3 = "[    1.  100.   10. 9999. ]";
    CHECK(example3.toString() == expected3);

    ty::Tensor example4(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}), std::vector<size_t>({2, 1, 2}),
                        {ty::DeviceType::CUDA, 0});

    const std::string expected4 = "[[[ 1. 2. ]]\n\n [[ 3. 4. ]]]";
    CHECK(example4.toString() == expected4);

    ty::Tensor example5(std::vector<int32_t>({1, 2, 3, 4}), std::vector<size_t>({2, 2, 1}), {ty::DeviceType::CUDA, 0},
                        ty::DType::Int32);

    const std::string expected5 = "[[[ 1 ]\n  [ 2 ]]\n\n [[ 3 ]\n  [ 4 ]]]";
    CHECK(example5.toString() == expected5);

    ty::Tensor example6(std::vector<int8_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
                        std::vector<size_t>({7, 7}), {ty::DeviceType::CUDA, 0}, ty::DType::Int8);

    const std::string expected6 = "[[ 1 1 1 ... 1 1 1 ]\n [ 1 1 1 ... 1 1 1 ]\n [ 1 1 1 ... 1 1 1 "
                                  "]\n ...\n "
                                  "[ 1 1 1 ... 1 1 1 ]\n [ 1 1 1 ... 1 1 1 ]\n [ 1 1 1 ... 1 1 1 ]]";
    CHECK(example6.toString() == expected6);
}