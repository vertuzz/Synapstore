//! BERT WordPiece tokenizer, matching Hugging Face `BertNormalizer` +
//! `BertPreTokenizer` + `WordPiece` token-for-token.

use super::unicode::{is_cjk, is_control, is_punctuation, is_whitespace, push_stripped};
use crate::error::{Error, Result};
use std::collections::HashMap;

/// Normalizer switches (from `tokenizer.json` / `tokenizer_config.json`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normalization {
    pub lowercase: bool,
    pub strip_accents: bool,
    pub chinese_chars: bool,
}

impl Default for Normalization {
    fn default() -> Self {
        Normalization { lowercase: true, strip_accents: true, chinese_chars: true }
    }
}

#[derive(Debug, Clone)]
pub struct WordPiece {
    vocab: HashMap<String, u32>,
    pub norm: Normalization,
    pub cls: u32,
    pub sep: u32,
    pub unk: u32,
    max_word_chars: usize,
}

impl WordPiece {
    /// Build from the lines of a `vocab.txt` (token id = line number).
    pub fn from_vocab(vocab_txt: &str, norm: Normalization) -> Result<Self> {
        let vocab = vocab_txt.lines().enumerate().map(|(i, t)| (t.trim_end_matches('\r').to_string(), i as u32));
        Self::from_map(vocab.collect(), norm)
    }

    /// Build from an explicit token → id map (e.g. `tokenizer.json`).
    pub fn from_map(vocab: HashMap<String, u32>, norm: Normalization) -> Result<Self> {
        let id = |t: &str| vocab.get(t).copied().ok_or_else(|| Error::InvalidInput(format!("vocab lacks {t}")));
        Ok(WordPiece { cls: id("[CLS]")?, sep: id("[SEP]")?, unk: id("[UNK]")?, vocab, norm, max_word_chars: 100 })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Normalize: clean control chars, space out CJK, strip accents, lowercase.
    pub fn normalize(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len() + 8);
        let mut tmp = String::new();
        for c in text.chars() {
            if c == '\0' || c == '\u{fffd}' || is_control(c) {
                continue;
            }
            let c = if is_whitespace(c) { ' ' } else { c };
            let cjk = self.norm.chinese_chars && is_cjk(c);
            if cjk {
                out.push(' ');
            }
            tmp.clear();
            if self.norm.strip_accents {
                push_stripped(c, &mut tmp);
            } else {
                tmp.push(c);
            }
            for t in tmp.chars() {
                if self.norm.lowercase {
                    out.extend(t.to_lowercase());
                } else {
                    out.push(t);
                }
            }
            if cjk {
                out.push(' ');
            }
        }
        out
    }

    /// Whitespace split, then isolate every punctuation character.
    pub fn pre_tokenize(text: &str) -> Vec<&str> {
        let mut words = Vec::new();
        for w in text.split(is_whitespace).filter(|w| !w.is_empty()) {
            let mut start = 0;
            for (i, c) in w.char_indices() {
                if is_punctuation(c) {
                    if start < i {
                        words.push(&w[start..i]);
                    }
                    words.push(&w[i..i + c.len_utf8()]);
                    start = i + c.len_utf8();
                }
            }
            if start < w.len() {
                words.push(&w[start..]);
            }
        }
        words
    }

    /// Greedy longest-match-first sub-word split of one word.
    fn word_piece(&self, word: &str, out: &mut Vec<u32>) {
        if word.chars().count() > self.max_word_chars {
            out.push(self.unk);
            return;
        }
        let mark = out.len();
        let mut start = 0;
        let mut piece = String::new();
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while start < end {
                piece.clear();
                if start > 0 {
                    piece.push_str("##");
                }
                piece.push_str(&word[start..end]);
                if let Some(&id) = self.vocab.get(&piece) {
                    found = Some(id);
                    break;
                }
                end = word[..end].char_indices().next_back().map_or(start, |(i, _)| i);
            }
            match found {
                Some(id) => out.push(id),
                None => {
                    out.truncate(mark);
                    out.push(self.unk);
                    return;
                }
            }
            start = end;
        }
    }

    /// `[CLS] tokens… [SEP]`, truncated to `max_len` ids in total.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<u32> {
        let budget = max_len.max(2) - 2;
        let mut ids = vec![self.cls];
        let normalized = self.normalize(text);
        let mut buf = Vec::new();
        for w in Self::pre_tokenize(&normalized) {
            buf.clear();
            self.word_piece(w, &mut buf);
            ids.extend_from_slice(&buf);
            if ids.len() > budget {
                break;
            }
        }
        ids.truncate(budget + 1);
        ids.push(self.sep);
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> WordPiece {
        let vocab = "[PAD]\n[UNK]\n[CLS]\n[SEP]\nun\n##aff\n##able\nhello\n,\nworld\n!\ncafe\n数\n";
        WordPiece::from_vocab(vocab, Normalization::default()).unwrap()
    }

    #[test]
    fn encodes_like_bert() {
        let t = tok();
        assert_eq!(t.encode("Hello, WORLD!", 512), vec![2, 7, 8, 9, 10, 3]);
        assert_eq!(t.encode("unaffable", 512), vec![2, 4, 5, 6, 3]);
        assert_eq!(t.encode("unknownword", 512), vec![2, 1, 3]);
        assert_eq!(t.encode("Café\u{200b}数", 512), vec![2, 11, 12, 3]);
        assert_eq!(t.encode("hello hello hello", 4), vec![2, 7, 7, 3]);
        assert_eq!(t.encode("", 512), vec![2, 3]);
    }
}
