//! Dense `C = A·W + bias` for transformer inference. Weights are packed
//! once into 16-column strips (`[strip][k][16]`) so the micro-kernel reads
//! them contiguously; AVX2+FMA is selected at runtime, other targets use a
//! portable kernel the compiler vectorizes (NEON on aarch64).

const COLS: usize = 16;
const ROWS: usize = 6;

/// A `[k][n]` matrix packed into zero-padded 16-column strips.
#[derive(Debug, Clone)]
pub struct Packed {
    pub k: usize,
    pub n: usize,
    data: Vec<f32>,
}

impl Packed {
    /// Pack with `get(p, j)` returning element (row p of k, column j of n).
    fn build(k: usize, n: usize, get: impl Fn(usize, usize) -> f32) -> Self {
        let strips = n.div_ceil(COLS);
        let mut data = vec![0f32; strips * k * COLS];
        for s in 0..strips {
            for p in 0..k {
                for q in 0..COLS.min(n - s * COLS) {
                    data[(s * k + p) * COLS + q] = get(p, s * COLS + q);
                }
            }
        }
        Packed { k, n, data }
    }
    /// From a row-major `[k][n]` matrix.
    pub fn from_rows(m: &[f32], k: usize, n: usize) -> Self {
        assert_eq!(m.len(), k * n, "pack shape");
        Self::build(k, n, |p, j| m[p * n + j])
    }
    /// From a PyTorch `Linear` weight `[out][in]` (computes `x·Wᵀ`).
    pub fn from_linear(w: &[f32], out: usize, inp: usize) -> Self {
        assert_eq!(w.len(), out * inp, "pack shape");
        Self::build(inp, out, |p, j| w[j * inp + p])
    }
    fn strip(&self, s: usize) -> &[f32] {
        &self.data[s * self.k * COLS..(s + 1) * self.k * COLS]
    }
}

#[inline(always)]
fn store<const R: usize>(c: &mut [f32], acc: &[[f32; COLS]; R], i: usize, j0: usize, n: usize) {
    let w = COLS.min(n - j0);
    for (r, row) in acc.iter().enumerate() {
        c[(i + r) * n + j0..(i + r) * n + j0 + w].copy_from_slice(&row[..w]);
    }
}

#[inline(always)]
fn tile_generic<const R: usize>(a: &[f32], w: &Packed, bias: &[f32; COLS], c: &mut [f32], i: usize, s: usize) {
    let k = w.k;
    let mut acc = [*bias; R];
    for (p, wt) in w.strip(s).as_chunks::<COLS>().0.iter().enumerate() {
        for (r, acc_r) in acc.iter_mut().enumerate() {
            let x = a[(i + r) * k + p];
            for q in 0..COLS {
                acc_r[q] =
                    if cfg!(target_arch = "aarch64") { x.mul_add(wt[q], acc_r[q]) } else { x * wt[q] + acc_r[q] };
            }
        }
    }
    store::<R>(c, &acc, i, s * COLS, w.n);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn tile_avx2<const R: usize>(a: &[f32], w: &Packed, bias: &[f32; COLS], c: &mut [f32], i: usize, s: usize) {
    use std::arch::x86_64::*;
    let k = w.k;
    let strip = w.strip(s);
    assert!(a.len() >= (i + R) * k && strip.len() == k * COLS);
    let mut out = [[0f32; COLS]; R];
    // SAFETY: rows i..i+R of `a` and the k×16 strip are in bounds (asserted).
    unsafe {
        let b = [_mm256_loadu_ps(bias.as_ptr()), _mm256_loadu_ps(bias.as_ptr().add(8))];
        let mut acc = [b; R];
        let ap = a.as_ptr().add(i * k);
        let mut wp = strip.as_ptr();
        for p in 0..k {
            let w0 = _mm256_loadu_ps(wp);
            let w1 = _mm256_loadu_ps(wp.add(8));
            for (r, acc_r) in acc.iter_mut().enumerate() {
                let x = _mm256_broadcast_ss(&*ap.add(r * k + p));
                acc_r[0] = _mm256_fmadd_ps(x, w0, acc_r[0]);
                acc_r[1] = _mm256_fmadd_ps(x, w1, acc_r[1]);
            }
            wp = wp.add(COLS);
        }
        for (o, acc_r) in out.iter_mut().zip(&acc) {
            _mm256_storeu_ps(o.as_mut_ptr(), acc_r[0]);
            _mm256_storeu_ps(o.as_mut_ptr().add(8), acc_r[1]);
        }
    }
    store::<R>(c, &out, i, s * COLS, w.n);
}

macro_rules! dispatch {
    ($tile:ident, $rows:expr, $($args:expr),*) => {
        match $rows {
            6 => $tile::<6>($($args),*),
            5 => $tile::<5>($($args),*),
            4 => $tile::<4>($($args),*),
            3 => $tile::<3>($($args),*),
            2 => $tile::<2>($($args),*),
            _ => $tile::<1>($($args),*),
        }
    };
}

/// `a` is `[m][w.k]`, `bias` is `[w.n]`; returns `[m][w.n]`.
pub fn linear(a: &[f32], w: &Packed, bias: &[f32]) -> Vec<f32> {
    let (k, n) = (w.k, w.n);
    assert!(k > 0 && a.len().is_multiple_of(k) && bias.len() == n, "gemm shape mismatch");
    let m = a.len() / k;
    let mut c = vec![0f32; m * n];
    #[cfg(target_arch = "x86_64")]
    let simd = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let simd = false;
    // Strips outermost: one k×16 strip stays in L1 while all rows stream by.
    for s in 0..n.div_ceil(COLS) {
        let mut b = [0f32; COLS];
        let j0 = s * COLS;
        b[..COLS.min(n - j0)].copy_from_slice(&bias[j0..(j0 + COLS).min(n)]);
        let mut i = 0;
        while i < m {
            let rows = ROWS.min(m - i);
            if simd {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: AVX2 and FMA were detected at runtime.
                unsafe {
                    dispatch!(tile_avx2, rows, a, w, &b, &mut c, i, s)
                }
            } else {
                dispatch!(tile_generic, rows, a, w, &b, &mut c, i, s)
            }
            i += rows;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_naive() {
        for (m, k, n) in [(1, 3, 2), (5, 7, 33), (9, 16, 48), (4, 384, 16), (13, 20, 35), (6, 8, 16), (7, 5, 1)] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i * 37 % 17) as f32 - 8.0) / 7.0).collect();
            let w: Vec<f32> = (0..k * n).map(|i| ((i * 13 % 11) as f32 - 5.0) / 3.0).collect();
            let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let c = linear(&a, &Packed::from_rows(&w, k, n), &b);
            // Same product through the Linear ([out][in]) packing.
            let wt: Vec<f32> = (0..n * k).map(|i| w[(i % k) * n + i / k]).collect();
            assert_eq!(c, linear(&a, &Packed::from_linear(&wt, n, k), &b));
            for i in 0..m {
                for j in 0..n {
                    let want: f32 = b[j] + (0..k).map(|p| a[i * k + p] * w[p * n + j]).sum::<f32>();
                    assert!((c[i * n + j] - want).abs() < 1e-3, "{m}x{k}x{n} at {i},{j}");
                }
            }
        }
    }
}
