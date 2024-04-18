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