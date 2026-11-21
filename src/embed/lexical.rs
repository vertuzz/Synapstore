//! Built-in, dependency-free text embedder (feature hashing).
//!
//! SynapStore's contract is that agents pass their own embeddings. When an
//! MCP client cannot (e.g. Claude Code has no embedding endpoint), this
//! deterministic *lexical* embedder is used instead: word unigrams, bigrams
//! and character trigrams are hashed into a signed, L2-normalized vector.
//! It captures word overlap and fuzzy spelling, not deep semantics.

use crate::hash::xxh64;

/// Default dimension of built-in embeddings.
pub const BUILTIN_DIM: usize = 384;
/// Marker persisted in the store's metadata for built-in embeddings.
pub const SOURCE_BUILTIN: &str = "builtin-hash-v1";
/// Marker for caller-provided embeddings.
pub const SOURCE_EXTERNAL: &str = "external";

const SEED: u64 = 0x4558_5048_4153_4831;

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "this", "to", "was", "were", "will", "with",
];

/// Lowercased alphanumeric tokens of `text`.
pub fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()).map(str::to_lowercase).collect()
}

/// Crude suffix stripping so "works"/"worked"/"working" share a feature.
pub fn stem(t: &str) -> &str {
    for suf in ["ing", "ed", "es", "s"] {
        if t.len() > suf.len() + 2 && t.is_char_boundary(t.len() - suf.len()) && t.ends_with(suf) {
            return &t[..t.len() - suf.len()];
        }
    }
    t
}

fn add(v: &mut [f32], feature: &str, weight: f32) {
    let h = xxh64(feature.as_bytes(), SEED);
    let idx = (h % v.len() as u64) as usize;
    let sign = if h >> 63 == 1 { -1.0 } else { 1.0 };
    v[idx] += sign * weight;
}

/// Embed `text` into a unit vector of dimension `dim` (`dim >= 1`).
pub fn embed(text: &str, dim: usize) -> Vec<f32> {
    let dim = dim.max(1);
    let mut v = vec![0.0f32; dim];
    let toks = tokens(text);
    for (i, t) in toks.iter().enumerate() {
        let stop = STOPWORDS.contains(&t.as_str());
        add(&mut v, &format!("w:{t}"), if stop { 0.25 } else { 0.7 });
        if !stop {
            add(&mut v, &format!("s:{}", stem(t)), 1.0);
        }
        if let Some(next) = toks.get(i + 1) {
            add(&mut v, &format!("b:{t} {next}"), 0.5);
        }
        if !stop {
            let padded: Vec<char> = format!("^{t}$").chars().collect();
            for w in padded.windows(3) {
                let tri: String = w.iter().collect();
                add(&mut v, &format!("c:{tri}"), 0.2);
            }
        }
    }
    if toks.is_empty() {
        add(&mut v, &format!("raw:{}", text.trim()), 1.0);
    }
    crate::math::normalize(&v).unwrap_or_else(|_| {
        let mut e = vec![0.0; dim];
        e[0] = 1.0;
        e
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{dot, norm};

    #[test]
    fn deterministic_unit_vectors() {
        let a = embed("Alice works at Acme Corp", BUILTIN_DIM);
        assert_eq!(a, embed("Alice works at Acme Corp", BUILTIN_DIM));
        assert!((norm(&a) - 1.0).abs() < 1e-5);
        assert!((norm(&embed("", 16)) - 1.0).abs() < 1e-5);
        assert!((norm(&embed("!!!", 16)) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn stemming() {
        assert_eq!(stem("works"), "work");
        assert_eq!(stem("working"), "work");
        assert_eq!(stem("is"), "is");
        assert_eq!(stem("ünïs"), "ünï");
    }

    #[test]
    fn lexical_overlap_ranks_higher() {
        let q = embed("where does alice work", BUILTIN_DIM);
        let rel = embed("Alice works at Acme Corp as an engineer", BUILTIN_DIM);
        let unrel = embed("The weather in Paris was rainy yesterday", BUILTIN_DIM);
        assert!(dot(&q, &rel) > dot(&q, &unrel) + 0.1);
    }
}
