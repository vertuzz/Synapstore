//! Deterministic content hashing (PRD §6.3: ids are `hash(normalized_text)`).
//!
//! XXH64 is implemented from the public specification so ids are stable across
//! Rust versions and platforms (unlike `std`'s `DefaultHasher`).

use std::hash::{BuildHasherDefault, Hasher};

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

/// Namespace seeds keep ids of different record kinds disjoint.
pub const NS_CHUNK: u64 = 0x5359_4e41_5043_484b; // "SYNAPCHK"
pub const NS_ENTITY: u64 = 0x5359_4e41_5045_4e54; // "SYNAPENT"
pub const NS_PREDICATE: u64 = 0x5359_4e41_5050_5244; // "SYNAPPRD"
pub const NS_EVENT: u64 = 0x5359_4e41_5045_5654; // "SYNAPEVT"

#[inline]
fn read64(d: &[u8], i: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&d[i..i + 8]);
    u64::from_le_bytes(b)
}

#[inline]
fn read32(d: &[u8], i: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&d[i..i + 4]);
    u32::from_le_bytes(b)
}

#[inline]
fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(P2)).rotate_left(31).wrapping_mul(P1)
}

#[inline]
fn merge(acc: u64, val: u64) -> u64 {
    (acc ^ round(0, val)).wrapping_mul(P1).wrapping_add(P4)
}

/// XXH64 of `data` with `seed`.
pub fn xxh64(data: &[u8], seed: u64) -> u64 {
    let len = data.len();
    let mut i = 0;
    let mut h = if len >= 32 {
        let mut v = [seed.wrapping_add(P1).wrapping_add(P2), seed.wrapping_add(P2), seed, seed.wrapping_sub(P1)];
        while i + 32 <= len {
            for (k, acc) in v.iter_mut().enumerate() {
                *acc = round(*acc, read64(data, i + 8 * k));
            }
            i += 32;
        }
        let mut h = v[0]
            .rotate_left(1)
            .wrapping_add(v[1].rotate_left(7))
            .wrapping_add(v[2].rotate_left(12))
            .wrapping_add(v[3].rotate_left(18));
        for acc in v {
            h = merge(h, acc);
        }
        h
    } else {
        seed.wrapping_add(P5)
    };
    h = h.wrapping_add(len as u64);
    while i + 8 <= len {
        h ^= round(0, read64(data, i));
        h = h.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
        i += 8;
    }
    if i + 4 <= len {
        h ^= u64::from(read32(data, i)).wrapping_mul(P1);
        h = h.rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
        i += 4;
    }
    while i < len {
        h ^= u64::from(data[i]).wrapping_mul(P5);
        h = h.rotate_left(11).wrapping_mul(P1);
        i += 1;
    }
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

/// SplitMix64 finalizer: a cheap, high-quality bijective mixer.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Canonical text form used for identity: lowercase, whitespace collapsed.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for word in text.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        for c in word.chars() {
            out.extend(c.to_lowercase());
        }
    }
    out
}

/// Deterministic id for `text` in namespace `ns`. Never returns 0, which is
/// reserved as the "none" sentinel in records.
pub fn content_id(ns: u64, text: &str) -> u64 {
    match xxh64(normalize(text).as_bytes(), ns) {
        0 => 1,
        h => h,
    }
}

pub fn chunk_id(text: &str) -> u64 {
    content_id(NS_CHUNK, text)
}

pub fn entity_id(name: &str) -> u64 {
    content_id(NS_ENTITY, name)
}

pub fn predicate_id(name: &str) -> u64 {
    content_id(NS_PREDICATE, name)
}

/// Format an id the way it is exposed over MCP (JSON numbers lose precision
/// above 2^53, so ids travel as 16-digit hex strings).
pub fn id_to_hex(id: u64) -> String {
    format!("{id:016x}")
}

/// Parse an id produced by [`id_to_hex`] (an optional `0x` prefix is accepted).
pub fn hex_to_id(s: &str) -> Option<u64> {
    let t = s.trim();
    let t = t.strip_prefix("0x").unwrap_or(t);
    if t.is_empty() || t.len() > 16 {
        return None;
    }
    u64::from_str_radix(t, 16).ok()
}

/// Hasher for keys that are already well-distributed u64 ids.
#[derive(Default, Clone, Copy)]
pub struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0 = xxh64(bytes, self.0);
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = mix64(self.0 ^ v);
    }
}

pub type IdBuild = BuildHasherDefault<IdHasher>;
pub type IdSet = std::collections::HashSet<u64, IdBuild>;
pub type IdMap<V> = std::collections::HashMap<u64, V, IdBuild>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xxh64_reference_vectors() {
        assert_eq!(xxh64(b"", 0), 0xEF46_DB37_51D8_E999);
        assert_eq!(xxh64(b"a", 0), 0xD24E_C4F1_A98C_6E5B);
        assert_eq!(xxh64(b"abc", 0), 0x44BC_2CF5_AD77_0999);
        assert_eq!(xxh64(b"Nobody inspects the spammish repetition", 0), 0xFBCE_A83C_8A37_8BF1);
    }

    #[test]
    fn normalization_makes_ids_stable() {
        assert_eq!(chunk_id("Hello   World\n"), chunk_id("hello world"));
        assert_ne!(chunk_id("hello world"), entity_id("hello world"));
        assert_eq!(normalize("  ÄBC  def "), "äbc def");
    }

    #[test]
    fn hex_roundtrip() {
        for id in [1u64, 42, u64::MAX, 0x0123_4567_89ab_cdef] {
            assert_eq!(hex_to_id(&id_to_hex(id)), Some(id));
        }
        assert_eq!(hex_to_id("0x10"), Some(16));
        assert_eq!(hex_to_id("zz"), None);
        assert_eq!(hex_to_id("11112222333344445"), None);
    }
}
