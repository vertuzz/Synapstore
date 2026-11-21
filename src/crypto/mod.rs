//! Small, dependency-free crypto helpers: SHA-256, OS randomness, hex and
//! base64url encoding, constant-time comparison.

pub mod sha256;

pub use sha256::{Sha256, sha256};

use crate::error::{Error, Result};

/// Lowercase hex encoding.
pub fn hex(bytes: &[u8]) -> String {
    const D: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(D[(b >> 4) as usize] as char);
        s.push(D[(b & 15) as usize] as char);
    }
    s
}

/// Decode lowercase/uppercase hex; `None` on malformed input.
pub fn unhex(s: &str) -> Option<Vec<u8>> {
    let b = s.as_bytes();
    if !b.len().is_multiple_of(2) {
        return None;
    }
    let nib = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
    b.chunks_exact(2).map(|p| Some(nib(p[0])? << 4 | nib(p[1])?)).collect()
}

/// Unpadded base64url (RFC 4648 §5).
pub fn base64url(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..=c.len() {
            s.push(A[(n >> (18 - 6 * i) & 63) as usize] as char);
        }
    }
    s
}

/// Decode standard or URL-safe base64 (padding optional); `None` if malformed.
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' | b'-' => Some(62),
        b'/' | b'_' => Some(63),
        _ => None,
    };
    let bytes: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let data = bytes.strip_suffix(b"==").or_else(|| bytes.strip_suffix(b"=")).unwrap_or(&bytes);
    let mut out = Vec::with_capacity(data.len() * 3 / 4);
    for chunk in data.chunks(4) {
        if chunk.len() == 1 {
            return None;
        }
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= (val(c)? as u32) << (18 - 6 * i);
        }
        out.extend_from_slice(&n.to_be_bytes()[1..chunk.len()]);
    }
    Some(out)
}

/// Compare two byte strings in time independent of where they differ.
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let diff = a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y));
    std::hint::black_box(diff) == 0
}

/// Fill `buf` from the operating system's CSPRNG.
pub fn random_bytes(buf: &mut [u8]) -> Result<()> {
    os_random(buf).map_err(|e| Error::Storage(format!("OS random generator unavailable: {e}")))
}

#[cfg(unix)]
fn os_random(buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::Read;
    std::fs::File::open("/dev/urandom")?.read_exact(buf)
}

#[cfg(windows)]
fn os_random(buf: &mut [u8]) -> std::io::Result<()> {
    #[link(name = "advapi32")]
    unsafe extern "system" {
        #[link_name = "SystemFunction036"]
        fn rtl_gen_random(buf: *mut u8, len: u32) -> u8;
    }
    for chunk in buf.chunks_mut(u32::MAX as usize) {
        // SAFETY: the pointer/length pair describes a valid, writable slice.
        if unsafe { rtl_gen_random(chunk.as_mut_ptr(), chunk.len() as u32) } == 0 {
            return Err(std::io::Error::other("RtlGenRandom failed"));
        }
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn os_random(_buf: &mut [u8]) -> std::io::Result<()> {
    Err(std::io::Error::other("no OS random source on this platform"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodings() {
        assert_eq!(hex(&[0, 15, 255]), "000fff");
        assert_eq!(unhex("000fFF").unwrap(), vec![0, 15, 255]);
        assert!(unhex("abc").is_none() && unhex("zz").is_none());
        assert_eq!(base64url(b""), "");
        assert_eq!(base64url(b"f"), "Zg");
        assert_eq!(base64url(b"fo"), "Zm8");
        assert_eq!(base64url(b"foo"), "Zm9v");
        assert_eq!(base64url(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64url(&[0xfb, 0xff]), "-_8");
        for raw in [&b""[..], b"f", b"fo", b"foo", b"foobar", &[0xfb, 0xff, 0x00]] {
            assert_eq!(base64_decode(&base64url(raw)).unwrap(), raw);
        }
        assert_eq!(base64_decode("Zm9vYg==").unwrap(), b"foob");
        assert_eq!(base64_decode("+/8=").unwrap(), vec![0xfb, 0xff]);
        assert!(base64_decode("Z").is_none() && base64_decode("Z!==").is_none());
    }

    #[test]
    fn randomness_and_ct_eq() {
        let (mut a, mut b) = ([0u8; 32], [0u8; 32]);
        random_bytes(&mut a).unwrap();
        random_bytes(&mut b).unwrap();
        assert_ne!(a, b);
        assert!(ct_eq(&a, &a) && !ct_eq(&a, &b) && !ct_eq(&a, &a[..31]));
    }
}
