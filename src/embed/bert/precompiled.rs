//! SentencePiece's "precompiled charsmap" normalizer (an NFKC-style
//! mapping stored as a Darts double-array trie plus a string pool), as used
//! by XLM-R / multilingual-E5 tokenizers.

use super::unicode::is_mark;
use crate::error::{Error, Result};

pub struct Precompiled {
    trie: Vec<u32>,
    pool: Vec<u8>,
}

impl Precompiled {
    /// From the raw (base64-decoded) `precompiled_charsmap` blob.
    pub fn from_blob(blob: &[u8]) -> Result<Self> {
        let bad = || Error::InvalidInput("tokenizer.json: malformed precompiled_charsmap".into());
        let size = u32::from_le_bytes(blob.get(..4).ok_or_else(bad)?.try_into().map_err(|_| bad())?) as usize;
        let trie_bytes = blob.get(4..4 + size).ok_or_else(bad)?;
        let trie = trie_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(Precompiled { trie, pool: blob[4 + size..].to_vec() })
    }

    /// Darts common-prefix search; returns the first (shortest) match.
    fn lookup(&self, key: &[u8]) -> Option<u32> {
        let unit = |p: usize| self.trie.get(p).copied();
        let offset = |u: u32| ((u >> 10) << ((u & (1 << 9)) >> 6)) as usize;
        let mut pos = offset(unit(0)?);
        for &c in key {
            if c == 0 {
                break;
            }
            pos ^= c as usize;
            let u = unit(pos)?;
            if u & ((1 << 31) | 0xFF) != c as u32 {
                return None;
            }
            pos ^= offset(u);
            if (u >> 8) & 1 == 1 {
                return Some(unit(pos)? & ((1 << 31) - 1));
            }
        }
        None
    }

    fn transform(&self, s: &str) -> Option<&str> {
        let start = self.lookup(s.as_bytes())? as usize;
        let rest = self.pool.get(start..)?;
        let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
        std::str::from_utf8(&rest[..end]).ok()
    }

    /// Normalize `text`, cluster by cluster (base char + combining marks /
    /// joiners approximate grapheme clusters), falling back to per char.
    pub fn normalize(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut i = 0;
        while i < chars.len() {
            let mut j = i + 1;
            while j < chars.len() {
                let (c, prev) = (chars[j].1, chars[j - 1].1);
                if is_mark(c) || matches!(c, '\u{200d}' | '\u{fe00}'..='\u{fe0f}') || prev == '\u{200d}' {
                    j += 1;
                } else {
                    break;
                }
            }
            let end = chars.get(j).map_or(text.len(), |c| c.0);
            let cluster = &text[chars[i].0..end];
            match (cluster.len() < 6).then(|| self.transform(cluster)).flatten() {
                Some(n) => out.push_str(n),
                None => {
                    for (k, c) in cluster.char_indices() {
                        match self.transform(&cluster[k..k + c.len_utf8()]) {
                            Some(n) => out.push_str(n),
                            None => out.push(c),
                        }
                    }
                }
            }
            i = j;
        }
        out
    }
}
