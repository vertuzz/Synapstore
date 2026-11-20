use super::*;
use crate::synth::rng::Rng;

fn reference(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let mut r = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        r.0 += x * y;
        r.1 += x * x;
        r.2 += y * y;
    }
    r
}

fn close(got: f32, want: f64, scale: f64) -> bool {
    (f64::from(got) - want).abs() <= 1e-4 * scale.max(1.0)
}

const DIMS: [usize; 15] = [1, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 384, 768, 1536, 1537];

#[test]
fn all_kernels_match_f64_reference() {
    let mut rng = Rng::new(1);
    for &d in &DIMS {
        let a: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
        let b: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
        let (ab, aa, bb) = reference(&a, &b);
        let scale = (aa * bb).sqrt();
        assert!(close(dot(&a, &b), ab, scale), "dispatch dot d={d}");
        // SAFETY: slices are valid for `d` reads.
        unsafe {
            assert!(close(scalar::dot_raw(a.as_ptr(), b.as_ptr(), d), ab, scale));
            let (x, y, z) = scalar::dot3_raw(a.as_ptr(), b.as_ptr(), d);
            assert!(close(x, ab, scale) && close(y, aa, aa) && close(z, bb, bb));
            #[cfg(target_arch = "x86_64")]
            {
                assert!(close(x86::dot_sse(a.as_ptr(), b.as_ptr(), d), ab, scale));
                let (x, y, z) = x86::dot3_sse(a.as_ptr(), b.as_ptr(), d);
                assert!(close(x, ab, scale) && close(y, aa, aa) && close(z, bb, bb));
                if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                    assert!(close(x86::dot_avx2(a.as_ptr(), b.as_ptr(), d), ab, scale));
                    let (x, y, z) = x86::dot3_avx2(a.as_ptr(), b.as_ptr(), d);
                    assert!(close(x, ab, scale) && close(y, aa, aa) && close(z, bb, bb));
                }
            }
        }
        let want_cos = ab / scale;
        assert!((f64::from(cosine(&a, &b)) - want_cos).abs() < 1e-4, "cosine d={d}");
    }
}

#[test]
fn byte_kernels_work_at_every_alignment() {
    let mut rng = Rng::new(2);
    for &d in &[384usize, 768, 1536, 17] {
        let a: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
        let b: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
        let want = f64::from(dot(&a, &b));
        let bytes = crate::archive::f32s_to_le(&b);
        let abytes = crate::archive::f32s_to_le(&a);
        for off in 0..4 {
            let mut buf = vec![0u8; off];
            buf.extend_from_slice(&bytes);
            let mut abuf = vec![0u8; 3 - off];
            abuf.extend_from_slice(&abytes);
            assert!(close(dot_le(&a, &buf[off..]), want, want.abs()));
            assert!(close(dot_le_le(&abuf[3 - off..], &buf[off..]), want, want.abs()));
        }
    }
}

#[test]
fn cosine_edge_cases() {
    assert_eq!(cosine(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
    assert_eq!(cosine(&[], &[]), 0.0);
    let v = [0.3f32, -0.4, 0.5];
    assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    let neg: Vec<f32> = v.iter().map(|x| -x).collect();
    assert!((cosine(&v, &neg) + 1.0).abs() < 1e-6);
}

#[test]
fn normalize_validates() {
    assert!(normalize(&[]).is_err());
    assert!(normalize(&[0.0, 0.0]).is_err());
    assert!(normalize(&[1.0, f32::NAN]).is_err());
    assert!(normalize(&[f32::INFINITY]).is_err());
    let big = normalize(&[f32::MAX, f32::MAX]).unwrap();
    assert!((norm(&big) - 1.0).abs() < 1e-5);
    let n = normalize(&[3.0, 4.0]).unwrap();
    assert!((n[0] - 0.6).abs() < 1e-6 && (n[1] - 0.8).abs() < 1e-6);
}

#[test]
fn backend_is_reported() {
    assert!(!backend().is_empty());
}
