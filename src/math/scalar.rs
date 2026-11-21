//! Portable scalar kernels. Eight independent accumulators let LLVM
//! auto-vectorize and hide FP latency on targets without explicit SIMD.

/// # Safety
/// `a` and `b` must be valid for `n` (possibly unaligned) f32 reads.
pub unsafe fn dot_raw(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut i = 0;
    unsafe {
        while i + 8 <= n {
            for (k, s) in acc.iter_mut().enumerate() {
                *s += a.add(i + k).read_unaligned() * b.add(i + k).read_unaligned();
            }
            i += 8;
        }
        let mut sum: f32 = acc.iter().sum();
        while i < n {
            sum += a.add(i).read_unaligned() * b.add(i).read_unaligned();
            i += 1;
        }
        sum
    }
}

/// # Safety
/// `a` and `b` must be valid for `n` (possibly unaligned) f32 reads.
pub unsafe fn dot3_raw(a: *const f32, b: *const f32, n: usize) -> (f32, f32, f32) {
    let (mut ab, mut aa, mut bb) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
    let mut i = 0;
    unsafe {
        while i + 4 <= n {
            for k in 0..4 {
                let x = a.add(i + k).read_unaligned();
                let y = b.add(i + k).read_unaligned();
                ab[k] += x * y;
                aa[k] += x * x;
                bb[k] += y * y;
            }
            i += 4;
        }
        let (mut s_ab, mut s_aa, mut s_bb) = (ab.iter().sum::<f32>(), aa.iter().sum::<f32>(), bb.iter().sum::<f32>());
        while i < n {
            let x = a.add(i).read_unaligned();
            let y = b.add(i).read_unaligned();
            s_ab += x * y;
            s_aa += x * x;
            s_bb += y * y;
            i += 1;
        }
        (s_ab, s_aa, s_bb)
    }
}

/// Endianness-independent dot of `a` with little-endian bytes `b`.
pub fn dot_le_portable(a: &[f32], b: &[u8]) -> f32 {
    a.iter().zip(b.as_chunks::<4>().0).map(|(x, c)| x * f32::from_le_bytes(*c)).sum()
}
