//! x86_64 kernels: AVX2+FMA (32 floats / iteration, 4 accumulators) and an
//! SSE2 baseline. All loads are `loadu` (alignment-agnostic).

use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    hsum128(_mm_add_ps(lo, hi))
}

#[inline]
fn hsum128(v: __m128) -> f32 {
    // SAFETY: SSE/SSE2 are part of the x86_64 baseline.
    unsafe {
        let shuf = _mm_movehl_ps(v, v);
        let sums = _mm_add_ps(v, shuf);
        let shuf = _mm_shuffle_ps::<0b01>(sums, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf))
    }
}

/// # Safety
/// CPU must support AVX2+FMA; `a`/`b` valid for `n` unaligned f32 reads.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_avx2(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    unsafe {
        while i + 32 <= n {
            for (k, s) in acc.iter_mut().enumerate() {
                let x = _mm256_loadu_ps(a.add(i + 8 * k));
                let y = _mm256_loadu_ps(b.add(i + 8 * k));
                *s = _mm256_fmadd_ps(x, y, *s);
            }
            i += 32;
        }
        while i + 8 <= n {
            acc[0] = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), acc[0]);
            i += 8;
        }
        let s = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let mut sum = hsum256(s);
        while i < n {
            sum += a.add(i).read_unaligned() * b.add(i).read_unaligned();
            i += 1;
        }
        sum
    }
}

/// # Safety
/// CPU must support AVX2+FMA; `a`/`b` valid for `n` unaligned f32 reads.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot3_avx2(a: *const f32, b: *const f32, n: usize) -> (f32, f32, f32) {
    let (mut ab, mut aa, mut bb) = ([_mm256_setzero_ps(); 2], [_mm256_setzero_ps(); 2], [_mm256_setzero_ps(); 2]);
    let mut i = 0;
    unsafe {
        while i + 16 <= n {
            for k in 0..2 {
                let x = _mm256_loadu_ps(a.add(i + 8 * k));
                let y = _mm256_loadu_ps(b.add(i + 8 * k));
                ab[k] = _mm256_fmadd_ps(x, y, ab[k]);
                aa[k] = _mm256_fmadd_ps(x, x, aa[k]);
                bb[k] = _mm256_fmadd_ps(y, y, bb[k]);
            }
            i += 16;
        }
        let (mut s_ab, mut s_aa, mut s_bb) = (
            hsum256(_mm256_add_ps(ab[0], ab[1])),
            hsum256(_mm256_add_ps(aa[0], aa[1])),
            hsum256(_mm256_add_ps(bb[0], bb[1])),
        );
        while i < n {
            let (x, y) = (a.add(i).read_unaligned(), b.add(i).read_unaligned());
            s_ab += x * y;
            s_aa += x * x;
            s_bb += y * y;
            i += 1;
        }
        (s_ab, s_aa, s_bb)
    }
}

/// # Safety
/// `a`/`b` valid for `n` unaligned f32 reads (SSE2 is baseline on x86_64).
pub unsafe fn dot_sse(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut i = 0;
    unsafe {
        let mut acc = [_mm_setzero_ps(); 4];
        while i + 16 <= n {
            for (k, s) in acc.iter_mut().enumerate() {
                let x = _mm_loadu_ps(a.add(i + 4 * k));
                let y = _mm_loadu_ps(b.add(i + 4 * k));
                *s = _mm_add_ps(*s, _mm_mul_ps(x, y));
            }
            i += 16;
        }
        let s = _mm_add_ps(_mm_add_ps(acc[0], acc[1]), _mm_add_ps(acc[2], acc[3]));
        let mut sum = hsum128(s);
        while i < n {
            sum += a.add(i).read_unaligned() * b.add(i).read_unaligned();
            i += 1;
        }
        sum
    }
}

/// # Safety
/// `a`/`b` valid for `n` unaligned f32 reads.
pub unsafe fn dot3_sse(a: *const f32, b: *const f32, n: usize) -> (f32, f32, f32) {
    let mut i = 0;
    unsafe {
        let (mut ab, mut aa, mut bb) = (_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps());
        while i + 4 <= n {
            let x = _mm_loadu_ps(a.add(i));
            let y = _mm_loadu_ps(b.add(i));
            ab = _mm_add_ps(ab, _mm_mul_ps(x, y));
            aa = _mm_add_ps(aa, _mm_mul_ps(x, x));
            bb = _mm_add_ps(bb, _mm_mul_ps(y, y));
            i += 4;
        }
        let (mut s_ab, mut s_aa, mut s_bb) = (hsum128(ab), hsum128(aa), hsum128(bb));
        while i < n {
            let (x, y) = (a.add(i).read_unaligned(), b.add(i).read_unaligned());
            s_ab += x * y;
            s_aa += x * x;
            s_bb += y * y;
            i += 1;
        }
        (s_ab, s_aa, s_bb)
    }
}
