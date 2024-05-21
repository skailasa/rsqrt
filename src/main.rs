use std::arch::x86_64::{__m256d, _mm256_mul_pd, _mm256_set1_pd};
use pulp::{f64x4};


use invsqrt::*;

fn main() {

    let simd = pulp::x86::V3::try_new().unwrap();
    let r2 = f64x4(1., 2., 3., 4.);

    let rinv = rqsqrt_approx_intrin_avx_64(pulp::cast(r2));


    let r2 = f64x4(1., 2., 3., 4.);
    let refined = rsqrt_double_intrin_64(pulp::cast(r2));

    let scale = unsafe { _mm256_set1_pd(1./16.) };
    let estimate = unsafe { _mm256_mul_pd(scale, refined) };
    println!("refined {:?}", estimate)
}