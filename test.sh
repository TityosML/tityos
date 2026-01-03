#!/usr/bin/env bash
set -e

CUDA_FLAG=OFF
AVX2_FLAG=OFF
CTEST_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --cuda)
      CUDA_FLAG=ON
      ;;
    --avx2)
      AVX2_FLAG=ON
      ;;
    --rerun-failed)
      CTEST_ARGS+=(--rerun-failed)
      ;;
    --output-on-failure)
      CTEST_ARGS+=(--output-on-failure)
      ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

# Building
cmake -S . -B build \
  -DTITYOS_BUILD_TESTS=ON \
  -DTITYOS_BUILD_WARNINGS=ON \
  -DTITYOS_USE_AVX2=${AVX2_FLAG} \
  -DTITYOS_USE_CUDA=${CUDA_FLAG}

cmake --build build --target tityos_tests -j

# Running Tests
cd build
ctest "${CTEST_ARGS[@]}"
cd ..
