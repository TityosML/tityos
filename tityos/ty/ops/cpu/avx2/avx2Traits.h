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
        static Vec mul(Vec a, Vec b) {
            __m256i aLower = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
            __m256i aUpper =
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));

            aLower = _mm256_mullo_epi16(aLower, b);
            aUpper = _mm256_mullo_epi16(aUpper, b);

            return _mm256_packs_epi16(aLower, aUpper);
        }

        static Vec empty() { return _mm256_setzero_si256(); }

        static int8_t sum(Vec a) {
            __m128i aLower = _mm256_castsi256_si128(a);
            __m128i aUpper = _mm256_extracti128_si256(a, 1);

            __m256i aLower16 = _mm256_cvtepi8_epi16(aLower);
            __m256i aUpper16 = _mm256_cvtepi8_epi16(aUpper);

            // Calculate Lower sum
            aLower = _mm256_castsi256_si128(aLower16);
            aUpper = _mm256_extracti128_si256(aLower16, 1);

            __m128i aLowerSum = _mm_hadd_epi16(aLower, aUpper);
            aLowerSum = _mm_hadd_epi16(aLowerSum, aLowerSum);
            aLowerSum = _mm_hadd_epi16(aLowerSum, aLowerSum);
            aLowerSum = _mm_hadd_epi16(aLowerSum, aLowerSum);

            // Calculate Upper sum
            aLower = _mm256_castsi256_si128(aUpper16);
            aUpper = _mm256_extracti128_si256(aUpper16, 1);

            __m128i aUpperSum = _mm_hadd_epi16(aLower, aUpper);
            aUpperSum = _mm_hadd_epi16(aUpperSum, aUpperSum);
            aUpperSum = _mm_hadd_epi16(aUpperSum, aUpperSum);
            aUpperSum = _mm_hadd_epi16(aUpperSum, aUpperSum);

            __m128i aSum = _mm_hadd_epi16(aLowerSum, aUpperSum);

            return static_cast<int8_t>(_mm_cvtsi128_si32(aSum));
        }
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
        static Vec mul(Vec a, Vec b) {
            __m256i aLower =
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 0));
            __m256i aUpper =
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));

            aLower = _mm256_mullo_epi16(aLower, b);
            aUpper = _mm256_mullo_epi16(aUpper, b);

            return _mm256_packs_epi16(aLower, aUpper);
        }

        static Vec empty() { return _mm256_setzero_si256(); }

        static uint8_t sum(Vec a) {
            __m128i aLower = _mm256_castsi256_si128(a);
            __m128i aUpper = _mm256_extracti128_si256(a, 1);

            __m256i aLower16 = _mm256_cvtepi8_epi16(aLower);
            __m256i aUpper16 = _mm256_cvtepi8_epi16(aUpper);

            // Calculate Lower sum
            aLower = _mm256_castsi256_si128(aLower16);
            aUpper = _mm256_extracti128_si256(aLower16, 1);

            __m128i aLowerSum = _mm_hadd_epi16(aLower, aUpper);
            aLowerSum = _mm_hadd_epi16(aLowerSum, aLowerSum);
            aLowerSum = _mm_hadd_epi16(aLowerSum, aLowerSum);
            aLowerSum = _mm_hadd_epi16(aLowerSum, aLowerSum);

            // Calculate Upper sum
            aLower = _mm256_castsi256_si128(aUpper16);
            aUpper = _mm256_extracti128_si256(aUpper16, 1);

            __m128i aUpperSum = _mm_hadd_epi16(aLower, aUpper);
            aUpperSum = _mm_hadd_epi16(aUpperSum, aUpperSum);
            aUpperSum = _mm_hadd_epi16(aUpperSum, aUpperSum);
            aUpperSum = _mm_hadd_epi16(aUpperSum, aUpperSum);

            __m128i aSum = _mm_hadd_epi16(aLowerSum, aUpperSum);

            return static_cast<uint8_t>(_mm_cvtsi128_si32(aSum));
        }
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
        static Vec mul(Vec a, Vec b) { return _mm256_mullo_epi16(a, b); }

        static Vec empty() { return _mm256_setzero_si256(); }

        static int16_t sum(Vec a) {
            __m128i aLower = _mm256_castsi256_si128(a);
            __m128i aUpper = _mm256_extracti128_si256(a, 1);

            __m128i aSum = _mm_hadd_epi16(aLower, aUpper);
            aSum = _mm_hadd_epi16(aSum, aSum);
            aSum = _mm_hadd_epi16(aSum, aSum);
            aSum = _mm_hadd_epi16(aSum, aSum);

            return static_cast<int16_t>(_mm_cvtsi128_si32(aSum));
        }
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
        static Vec mul(Vec a, Vec b) { return _mm256_mullo_epi16(a, b); }

        static Vec empty() { return _mm256_setzero_si256(); }

        static uint16_t sum(Vec a) {
            __m128i aLower = _mm256_castsi256_si128(a);
            __m128i aUpper = _mm256_extracti128_si256(a, 1);

            __m128i aSum = _mm_hadd_epi16(aLower, aUpper);
            aSum = _mm_hadd_epi16(aSum, aSum);
            aSum = _mm_hadd_epi16(aSum, aSum);
            aSum = _mm_hadd_epi16(aSum, aSum);

            return static_cast<uint16_t>(_mm_cvtsi128_si32(aSum));
        }
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
        static Vec mul(Vec a, Vec b) { return _mm256_mullo_epi32(a, b); }

        static Vec empty() { return _mm256_setzero_si256(); }

        static int32_t sum(Vec a) {
            __m128i aLower = _mm256_castsi256_si128(a);
            __m128i aUpper = _mm256_extracti128_si256(a, 1);

            __m128i aSum = _mm_hadd_epi32(aLower, aUpper);
            aSum = _mm_hadd_epi32(aSum, aSum);
            aSum = _mm_hadd_epi32(aSum, aSum);

            return _mm_cvtsi128_si32(aSum);
        }
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
        static Vec mul(Vec a, Vec b) { return _mm256_mullo_epi32(a, b); }

        static Vec empty() { return _mm256_setzero_si256(); }

        static uint32_t sum(Vec a) {
            __m128i aLower = _mm256_castsi256_si128(a);
            __m128i aUpper = _mm256_extracti128_si256(a, 1);

            __m128i aSum = _mm_hadd_epi32(aLower, aUpper);
            aSum = _mm_hadd_epi32(aSum, aSum);
            aSum = _mm_hadd_epi32(aSum, aSum);

            return _mm_cvtsi128_si32(aSum);
        }
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
        static Vec mul(Vec a, Vec b) {
            const __m256i mask32 = _mm256_set1_epi64x(0xFFFFFFFF);

            __m256i aLower = _mm256_and_si256(a, mask32);
            __m256i bLower = _mm256_and_si256(b, mask32);

            __m256i aUpper = _mm256_srli_epi64(a, 32);
            __m256i bUpper = _mm256_srli_epi64(b, 32);

            __m256i lower = _mm256_mul_epu32(aLower, bLower);

            __m256i cross1 = _mm256_mul_epu32(aLower, bUpper);
            __m256i cross2 = _mm256_mul_epu32(aUpper, bLower);

            __m256i cross = _mm256_add_epi64(cross1, cross2);
            cross = _mm256_slli_epi64(cross, 32);

            return _mm256_add_epi64(lower, cross);
        }

        static Vec empty() { return _mm256_setzero_si256(); }

        static int64_t sum(Vec a) {
            __m128i aLower = _mm256_castsi256_si128(a);
            __m128i aUpper = _mm256_extracti128_si256(a, 1);

            aLower = _mm_add_epi64(aLower, aUpper);

            __m128i tmp = _mm_unpackhi_epi64(aLower, aLower);
            aLower = _mm_add_epi64(aLower, tmp);

            return _mm_cvtsi128_si64(aLower);
        }
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
        static Vec mul(Vec a, Vec b) {
            const __m256i mask32 = _mm256_set1_epi64x(0xFFFFFFFF);

            __m256i aLower = _mm256_and_si256(a, mask32);
            __m256i bLower = _mm256_and_si256(b, mask32);

            __m256i aUpper = _mm256_srli_epi64(a, 32);
            __m256i bUpper = _mm256_srli_epi64(b, 32);

            __m256i lower = _mm256_mul_epu32(aLower, bLower);

            __m256i cross1 = _mm256_mul_epu32(aLower, bUpper);
            __m256i cross2 = _mm256_mul_epu32(aUpper, bLower);

            __m256i cross = _mm256_add_epi64(cross1, cross2);
            cross = _mm256_slli_epi64(cross, 32);

            return _mm256_add_epi64(lower, cross);
        }

        static Vec empty() { return _mm256_setzero_si256(); }

        static uint64_t sum(Vec a) {
            __m128i aLower = _mm256_castsi256_si128(a);
            __m128i aUpper = _mm256_extracti128_si256(a, 1);

            aLower = _mm_add_epi64(aLower, aUpper);

            __m128i tmp = _mm_unpackhi_epi64(aLower, aLower);
            aLower = _mm_add_epi64(aLower, tmp);

            return _mm_cvtsi128_si64(aLower);
        }
    };

    template <> struct Avx2Traits<float> {
        using Vec = __m256;
        static constexpr int lanes = 8;
        static Vec load(const float* p) { return _mm256_loadu_ps(p); }
        static void store(float* p, Vec v) { _mm256_storeu_ps(p, v); }
        static Vec add(Vec a, Vec b) { return _mm256_add_ps(a, b); }
        static Vec mul(Vec a, Vec b) { return _mm256_mul_ps(a, b); }

        static Vec empty() { return _mm256_setzero_ps(); }

        static float sum(Vec a) {
            __m128 aLower = _mm256_castps256_ps128(a);
            __m128 aUpper = _mm256_extractf128_ps(a, 1);

            __m128 aSum = _mm_hadd_ps(aLower, aUpper);
            aSum = _mm_hadd_ps(aSum, aSum);
            aSum = _mm_hadd_ps(aSum, aSum);

            return _mm_cvtss_f32(aSum);
        }
    };

    template <> struct Avx2Traits<double> {
        using Vec = __m256d;
        static constexpr int lanes = 4;
        static Vec load(const double* p) { return _mm256_loadu_pd(p); }
        static void store(double* p, Vec v) { _mm256_storeu_pd(p, v); }
        static Vec add(Vec a, Vec b) { return _mm256_add_pd(a, b); }
        static Vec mul(Vec a, Vec b) { return _mm256_mul_pd(a, b); }

        static Vec empty() { return _mm256_setzero_pd(); }

        static double sum(Vec a) {
            __m128d aLower = _mm256_castpd256_pd128(a);
            __m128d aUpper = _mm256_extractf128_pd(a, 1);

            __m128d aSum = _mm_hadd_pd(aLower, aUpper);
            aSum = _mm_hadd_pd(aSum, aSum);

            return _mm_cvtsd_f64(aSum);
        }
    };
}; // namespace internal
} // namespace ty