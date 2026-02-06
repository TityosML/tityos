#!/usr/bin/env bash
set -e

CUDA_FLAG=OFF
AVX2_FLAG=OFF
PERF_FLAG=OFF
CTEST_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --cuda)
      CUDA_FLAG=ON
      ;;
    --perf)
      PERF_FLAG=ON
      ;;
    --verbose)
      CTEST_ARGS+=(--verbose)
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
  -DTITYOS_BUILD_CUDA=${CUDA_FLAG} \
  -DTITYOS_ENABLE_PERFORMANCE_TESTS=${PERF_FLAG} 

cmake --build build --target tityos_cpu
cmake --build build --target tityos_tests -j

if [ "$CUDA_FLAG" = "ON" ]; then
  cmake --build build --target tityos_cuda -j
fi

# Running Tests
cd build
ctest "${CTEST_ARGS[@]}"
cd ..
