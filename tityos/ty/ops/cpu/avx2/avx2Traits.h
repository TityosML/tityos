#pragma once

#include "tityos/ty/tensor/Dtype.h"

#include <cstdint>
#include <immintrin.h>

namespace ty {
namespace internal {
    template <typename T> struct Avx2Traits;

    template <> struct Avx2Traits<int8_t> {
        using Vec = __m256i;
        static constexpr int lanes = 32;
        static Vec load(const int8_t* p) {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        }
        static void store(int8_t* p, Vec v) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
        }
        static Vec add(Vec a, Vec b) { return _mm256_add_epi8(a, b); }
    };

    template <> struct Avx2Traits<uint8_t> {
        using Vec = __m256i;
        static constexpr int lanes = 32;
        static Vec load(const uint8_t* p) {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        }
        static void store(uint8_t* p, Vec v) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
        }
        static Vec add(Vec a, Vec b) { return _mm256_add_epi8(a, b); }
    };

    template <> struct Avx2Traits<int16_t> {
        using Vec = __m256i;
        static constexpr int lanes = 16;
        static Vec load(const int16_t* p) {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        }
        static void store(int16_t* p, Vec v) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
        }
        static Vec add(Vec a, Vec b) { return _mm256_add_epi16(a, b); }
    };

    template <> struct Avx2Traits<uint16_t> {
        using Vec = __m256i;
        static constexpr int lanes = 16;
        static Vec load(const uint16_t* p) {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        }
        static void store(uint16_t* p, Vec v) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
        }
        static Vec add(Vec a, Vec b) { return _mm256_add_epi16(a, b); }
    };

    template <> struct Avx2Traits<int32_t> {
        using Vec = __m256i;
        static constexpr int lanes = 8;
        static Vec load(const int32_t* p) {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        }
        static void store(int32_t* p, Vec v) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
        }
        static Vec add(Vec a, Vec b) { return _mm256_add_epi32(a, b); }
    };

    template <> struct Avx2Traits<uint32_t> {
        using Vec = __m256i;
        static constexpr int lanes = 8;
        static Vec load(const uint32_t* p) {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        }
        static void store(uint32_t* p, Vec v) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
        }
        static Vec add(Vec a, Vec b) { return _mm256_add_epi32(a, b); }
    };

    template <> struct Avx2Traits<int64_t> {
        using Vec = __m256i;
        static constexpr int lanes = 4;
        static Vec load(const int64_t* p) {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        }
        static void store(int64_t* p, Vec v) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
        }
        static Vec add(Vec a, Vec b) { return _mm256_add_epi64(a, b); }
    };

    template <> struct Avx2Traits<uint64_t> {
        using Vec = __m256i;
        static constexpr int lanes = 4;
        static Vec load(const uint64_t* p) {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        }
        static void store(uint64_t* p, Vec v) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
        }
        static Vec add(Vec a, Vec b) { return _mm256_add_epi64(a, b); }
    };

    template <> struct Avx2Traits<float> {
        using Vec = __m256;
        static constexpr int lanes = 8;
        static Vec load(const float* p) { return _mm256_loadu_ps(p); }
        static void store(float* p, Vec v) { _mm256_storeu_ps(p, v); }
        static Vec add(Vec a, Vec b) { return _mm256_add_ps(a, b); }
    };

    template <> struct Avx2Traits<double> {
        using Vec = __m256d;
        static constexpr int lanes = 4;
        static Vec load(const double* p) { return _mm256_loadu_pd(p); }
        static void store(double* p, Vec v) { _mm256_storeu_pd(p, v); }
        static Vec add(Vec a, Vec b) { return _mm256_add_pd(a, b); }
    };
}; // namespace internal
} // namespace ty