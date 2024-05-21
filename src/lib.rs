use pulp::{f32x4, f32x8};
use std::arch::x86_64::{
    __m128, __m128d, __m256, __m256d, _mm256_andnot_ps, _mm256_cmp_ps, _mm256_cvtpd_ps,
    _mm256_cvtps_pd, _mm256_mul_pd, _mm256_mul_ps, _mm256_rsqrt_ps, _mm256_set1_pd, _mm256_set1_ps,
    _mm256_set_ps, _mm256_sub_pd, _mm256_sub_ps, _mm_andnot_ps, _mm_cmpeq_ps, _mm_rsqrt_ps,
    _CMP_EQ_OS,
};

/// Single precision rsqrt
pub fn rqsqrt_approx_intrin_avx_32(r2: __m256) -> __m256 {
    let zero = f32x8(0., 0., 0., 0., 0., 0., 0., 0.);
    unsafe {
        _mm256_andnot_ps(
            _mm256_cmp_ps(r2, pulp::cast(zero), _CMP_EQ_OS),
            _mm256_rsqrt_ps(r2),
        )
    }
}

/// Single precision rsqrt
pub fn rqsqrt_approx_intrin_sse_32(r2: __m128) -> __m128 {
    let zero = f32x4(0., 0., 0., 0.);
    unsafe { _mm_andnot_ps(_mm_cmpeq_ps(r2, pulp::cast(zero)), _mm_rsqrt_ps(r2)) }
}

/// Always apply single precision rsqrt instruction
pub fn rqsqrt_approx_intrin_avx_64(r2: __m256d) -> __m256d {
    unsafe { _mm256_cvtps_pd(rqsqrt_approx_intrin_sse_32(_mm256_cvtpd_ps(r2))) }
}

pub fn rsqrt_newton_intrin_32(rinv: __m256, r2: __m256, newton_constant: f32) -> __m256 {
    // Newton iteration rinv = 0.5 * (3 - r2 r_inv_appx^2)
    // defer scaling 0.5
    unsafe {
        let newton_constant = _mm256_set1_ps(newton_constant);
        _mm256_mul_ps(
            rinv,
            _mm256_sub_ps(
                newton_constant,
                _mm256_mul_ps(r2, _mm256_mul_ps(rinv, rinv)),
            ),
        )
    }
}

pub fn rsqrt_newton_intrin_64(rinv: __m256d, r2: __m256d, newton_constant: __m256d) -> __m256d {
    // Newton iteration rinv = 0.5 * (3 - r2 r_inv_appx^2)
    // defer scaling by 0.5

    unsafe {
        _mm256_mul_pd(
            rinv,
            _mm256_sub_pd(
                newton_constant,
                _mm256_mul_pd(r2, _mm256_mul_pd(rinv, rinv)),
            ),
        )
    }
}

pub fn rsqrt_single_intrin_32(r2: __m256) -> __m256 {
    // Approximate inverse
    let rinv = rqsqrt_approx_intrin_avx_32(r2);

    // Use as first guess in Newton step
    let rinv = rsqrt_newton_intrin_32(rinv, r2, 3.0);

    rinv
}

pub fn rsqrt_single_intrin_64(r2: __m256d) -> __m256d {
    // Approximate inverse
    let rinv = rqsqrt_approx_intrin_avx_64(r2);

    // Use as first guess in Newton step
    let newton_constant = unsafe { _mm256_set1_pd(3.0) };
    let rinv = rsqrt_newton_intrin_64(rinv, r2, newton_constant);

    // Still needs to be scaled by 1/2
    rinv
}

pub fn rsqrt_double_intrin_64(r2: __m256d) -> __m256d {
    // Approximate inverse
    let rinv = rqsqrt_approx_intrin_avx_64(r2);

    // Use as first guess in Newton step
    let newton_constant = unsafe { _mm256_set1_pd(3.0) };
    let rinv = rsqrt_newton_intrin_64(rinv, r2, newton_constant);

    let newton_constant = unsafe { _mm256_set1_pd(12.0) };
    let rinv = rsqrt_newton_intrin_64(rinv, r2, newton_constant);

    // Still needs to be scaled by 1/16
    rinv
}
