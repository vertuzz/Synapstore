//! Bespoke vector math (PRD §4, Layer 1): SIMD dot product / cosine similarity
//! with runtime dispatch (AVX2+FMA → SSE2 on x86_64, NEON on aarch64, portable
//! scalar elsewhere). Every kernel uses *unaligned* loads, so stored vectors
//! can be scored directly inside redb page buffers without copying.

pub mod gemm;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod scalar;
#[cfg(target_arch = "x86_64")]
pub mod x86;

use crate::error::{Error, Result};

/// Name of the kernel selected on this CPU (exported in metrics).
pub fn backend() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") { "avx2+fma" } else { "sse2" }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "neon"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "scalar"
    }
}

/// Raw kernels: `(dot, |a|², |b|²)` over `n` possibly-unaligned f32s.
///
/// # Safety
/// `a` and `b` must be valid for reads of `n` f32 values (alignment not required).
#[inline]
unsafe fn dot3_raw(a: *const f32, b: *const f32, n: usize) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { x86::dot3_avx2(a, b, n) };
        }
        unsafe { x86::dot3_sse(a, b, n) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::dot3_neon(a, b, n) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        unsafe { scalar::dot3_raw(a, b, n) }
    }
}

/// # Safety
/// Same contract as [`dot3_raw`].
#[inline]
unsafe fn dot_raw(a: *const f32, b: *const f32, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { x86::dot_avx2(a, b, n) };
        }
        unsafe { x86::dot_sse(a, b, n) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::dot_neon(a, b, n) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        unsafe { scalar::dot_raw(a, b, n) }
    }
}

/// Dot product of two f32 slices (over the shorter length).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    // SAFETY: both pointers are valid for `n` reads.
    unsafe { dot_raw(a.as_ptr(), b.as_ptr(), n) }
}

/// Dot product of `a` with a vector stored as little-endian f32 bytes at any
/// alignment (zero-copy scoring of stored embeddings).
pub fn dot_le(a: &[f32], b: &[u8]) -> f32 {
    let n = a.len().min(b.len() / 4);
    if cfg!(target_endian = "little") {
        // SAFETY: `b` holds at least `n * 4` bytes; kernels use unaligned loads.
        unsafe { dot_raw(a.as_ptr(), b.as_ptr().cast::<f32>(), n) }
    } else {
        scalar::dot_le_portable(a, b)
    }
}

/// Dot product of two vectors both stored as little-endian bytes.
pub fn dot_le_le(a: &[u8], b: &[u8]) -> f32 {
    let n = (a.len() / 4).min(b.len() / 4);
    if cfg!(target_endian = "little") {
        // SAFETY: both buffers hold at least `n * 4` bytes.
        unsafe { dot_raw(a.as_ptr().cast::<f32>(), b.as_ptr().cast::<f32>(), n) }
    } else {
        let av = crate::archive::le_to_f32s(a);
        scalar::dot_le_portable(&av, b)
    }
}

/// Cosine similarity in one SIMD pass. Returns 0.0 if either vector is zero.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    // SAFETY: both pointers are valid for `n` reads.
    let (ab, aa, bb) = unsafe { dot3_raw(a.as_ptr(), b.as_ptr(), n) };
    if aa <= 0.0 || bb <= 0.0 {
        return 0.0;
    }
    (ab / (aa.sqrt() * bb.sqrt())).clamp(-1.0, 1.0)
}

/// Euclidean norm.
pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// Validate a vector and scale it to unit length, so that cosine similarity
/// becomes a plain dot product inside the index.
pub fn normalize(v: &[f32]) -> Result<Vec<f32>> {
    if v.is_empty() {
        return Err(Error::InvalidVector("vector is empty".into()));
    }
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(Error::InvalidVector(format!("non-finite value at index {i}")));
    }
    // Accumulate in f64 so huge-but-finite inputs do not overflow to inf.
    let n = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    if n == 0.0 || !n.is_finite() {
        return Err(Error::InvalidVector("vector has zero (or non-finite) norm".into()));
    }
    Ok(v.iter().map(|x| (f64::from(*x) / n) as f32).collect())
}

#[cfg(test)]
mod tests;
