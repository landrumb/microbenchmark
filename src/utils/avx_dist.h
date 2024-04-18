#pragma once
#include <immintrin.h>

/* These pointers are assumed to be 64 bit aligned */
float sq_euclidean_aligned_100(const float* a, const float* b) {
    __m512 total = _mm512_setzero_ps();

    // we do 6 of these avx512 additions
    for (int i =0; i < 6; i++){
        __m512 a_chunk = _mm512_load_ps(a + 16*i);
        __m512 b_chunk = _mm512_load_ps(b + 16*i);

        a_chunk = _mm512_sub_ps(a_chunk, b_chunk);

        a_chunk = _mm512_mul_ps(a_chunk, a_chunk);

        total = _mm512_add_ps(total, a_chunk);
    }

    // we load and perform this operation on the remainder of 4 values
    __m128 a_remainder = _mm_load_ps(a + 96);
    __m128 b_remainder = _mm_load_ps(b + 96);

    a_remainder = _mm_sub_ps(a_remainder, b_remainder);

    a_remainder = _mm_mul_ps(a_remainder, a_remainder);

    total = _mm512_add_ps(total, _mm512_castps128_ps512(a_remainder));
    
    return _mm512_reduce_add_ps(total);
}

/* These pointers are NOT assumed to be 64 bit aligned */
float sq_euclidean_unaligned_100(const float* a, const float* b) {
    __m512 total = _mm512_setzero_ps();

    // we do 6 of these avx512 additions
    for (int i =0; i < 6; i++){
        __m512 a_chunk = _mm512_loadu_ps(a + 16*i);
        __m512 b_chunk = _mm512_loadu_ps(b + 16*i);

        a_chunk = _mm512_sub_ps(a_chunk, b_chunk);

        a_chunk = _mm512_mul_ps(a_chunk, a_chunk);

        total = _mm512_add_ps(total, a_chunk);
    }

    // we load and perform this operation on the remainder of 4 values
    __m128 a_remainder = _mm_loadu_ps(a + 96);
    __m128 b_remainder = _mm_loadu_ps(b + 96);

    a_remainder = _mm_sub_ps(a_remainder, b_remainder);

    a_remainder = _mm_mul_ps(a_remainder, a_remainder);

    total = _mm512_add_ps(total, _mm512_castps128_ps512(a_remainder));
    
    return _mm512_reduce_add_ps(total);
}

/* These pointers are assumed to be 64 bit aligned */
float sq_euclidean_aligned_pipeline2(const float* a, const float* b) {
    __m512 total1 = _mm512_setzero_ps();
    __m512 total2 = _mm512_setzero_ps();

    // we do 6 of these avx512 additions
    for (int i =0; i < 3; i++){

        __m512 a_chunk1 = _mm512_load_ps(a + 32*i);
        __m512 b_chunk1 = _mm512_load_ps(b + 32*i);

        __m512 a_chunk2 = _mm512_load_ps(a + 32*i + 16);
        __m512 b_chunk2 = _mm512_load_ps(b + 32*i + 16);
        
        a_chunk1 = _mm512_sub_ps(a_chunk1, b_chunk1);
        a_chunk1 = _mm512_mul_ps(a_chunk1, a_chunk1);

        a_chunk2 = _mm512_sub_ps(a_chunk2, b_chunk2);
        a_chunk2 = _mm512_mul_ps(a_chunk2, a_chunk2);

        total1 = _mm512_add_ps(total1, a_chunk1);

        total2 = _mm512_add_ps(total2, a_chunk2);
    }

    // we load and perform this operation on the remainder of 4 values
    __m128 a_remainder = _mm_load_ps(a + 96);
    __m128 b_remainder = _mm_load_ps(b + 96);

    a_remainder = _mm_sub_ps(a_remainder, b_remainder);
    a_remainder = _mm_mul_ps(a_remainder, a_remainder);

    total1 = _mm512_add_ps(total1, _mm512_castps128_ps512(a_remainder));

    total1 = _mm512_add_ps(total1, total2);
    
    return _mm512_reduce_add_ps(total1);
}

/* These pointers are NOT assumed to be 64 bit aligned 

This function assumes that AVX512F is available */
float sq_euclidean(const float* a, const float* b, size_t dim) {
    __m512 total = _mm512_setzero_ps();

    size_t n_512 = dim / 16;

    for (size_t i=0; i < n_512; i++){
        __m512 a_chunk = _mm512_loadu_ps(a + 16*i);
        __m512 b_chunk = _mm512_loadu_ps(b + 16*i);

        a_chunk = _mm512_sub_ps(a_chunk, b_chunk);
        a_chunk = _mm512_mul_ps(a_chunk, a_chunk);

        total = _mm512_add_ps(total, a_chunk);
    }

    bool remainder_256 = (dim % 16) >= 8;
    
    if (remainder_256) {
        __m256 a_chunk = _mm256_loadu_ps(a + 16 * n_512);
        __m256 b_chunk = _mm256_loadu_ps(b + 16 * n_512);

        a_chunk = _mm256_sub_ps(a_chunk, b_chunk);
        a_chunk = _mm256_mul_ps(a_chunk, a_chunk);

        total = _mm512_add_ps(total, _mm512_castps256_ps512(a_chunk));
    }

    bool remainder_128 = (dim % 16) % 8 >= 4;

    if (remainder_128) {
        __m128 a_chunk = _mm_load_ps(a + 16 * n_512 + (8 * remainder_256));
        __m128 b_chunk = _mm_load_ps(b + 16 * n_512 + (8 * remainder_256));

        a_chunk = _mm_sub_ps(a_chunk, b_chunk);
        a_chunk = _mm_mul_ps(a_chunk, a_chunk);

        total = _mm512_add_ps(total, _mm512_castps128_ps512(a_chunk));
    }

    // I mean realistically dimensionalities will be divisible by 4 so this is just for completeness
    size_t remainder = dim % 4;

    if (remainder) {
        // doing a masked load to get just what we care about
        // all values we want (first `remainder`) have 1 in the corresponding bit
        // incredibly, this is the inverse of a non-zeroing mask load
        __mmask8 mask = (1 << remainder) - 1;

        __m128 a_chunk = _mm_maskz_loadu_ps(mask, a + 16 * n_512 + (8 * remainder_256) + (4 * remainder_128));
        __m128 b_chunk = _mm_maskz_loadu_ps(mask, b + 16 * n_512 + (8 * remainder_256) + (4 * remainder_128));

        a_chunk = _mm_sub_ps(a_chunk, b_chunk);
        a_chunk = _mm_mul_ps(a_chunk, a_chunk);

        total = _mm512_add_ps(total, _mm512_castps128_ps512(a_chunk));
    }

    return _mm512_reduce_add_ps(total);
}

