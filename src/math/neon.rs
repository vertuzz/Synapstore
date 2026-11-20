//! aarch64 NEON kernels (NEON is baseline on aarch64). `vld1q_f32` performs
//! alignment-agnostic loads.

use std::arch::aarch64::*;

/// # Safety
/// `a`/`b` valid for `n` unaligned f32 reads.
pub unsafe fn dot_neon(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut i = 0;
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 4];
        while i + 16 <= n {
            for (k, s) in acc.iter_mut().enumerate() {
                *s = vfmaq_f32(*s, vld1q_f32(a.add(i + 4 * k)), vld1q_f32(b.add(i + 4 * k)));
            }
            i += 16;
        }
        let s = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        let mut sum = vaddvq_f32(s);
        while i < n {
            sum += a.add(i).read_unaligned() * b.add(i).read_unaligned();
            i += 1;
        }
        sum
    }
}

/// # Safety
/// `a`/`b` valid for `n` unaligned f32 reads.
pub unsafe fn dot3_neon(a: *const f32, b: *const f32, n: usize) -> (f32, f32, f32) {
    let mut i = 0;
    unsafe {
        let (mut ab, mut aa, mut bb) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        while i + 4 <= n {
            let x = vld1q_f32(a.add(i));
            let y = vld1q_f32(b.add(i));
            ab = vfmaq_f32(ab, x, y);
            aa = vfmaq_f32(aa, x, x);
            bb = vfmaq_f32(bb, y, y);
            i += 4;
        }
        let (mut s_ab, mut s_aa, mut s_bb) = (vaddvq_f32(ab), vaddvq_f32(aa), vaddvq_f32(bb));
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
