//! SentencePiece Unigram tokenizer (XLM-R / multilingual-E5 style), matching
//! Hugging Face `tokenizers`: Precompiled normalizer, Metaspace
//! pre-tokenizer, Viterbi segmentation and `<s> … </s>` post-processing.

use super::precompiled::Precompiled;
use crate::error::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;

const UNK_PENALTY: f32 = 10.0;
const SPACE: char = '\u{2581}';

pub struct Unigram {
    pieces: HashMap<String, (u32, f32)>,
    max_chars: usize,
    unk: u32,
    unk_score: f32,
    pub cls: u32,
    pub sep: u32,
    precompiled: Option<Precompiled>,
    collapse_spaces: bool,
    lowercase: bool,
    prefix_space: bool,
}

fn bad(msg: &str) -> Error {
    Error::InvalidInput(format!("tokenizer.json: {msg}"))
}

impl Unigram {
    pub fn from_json(tj: &Value) -> Result<Self> {
        let model = &tj["model"];
        let vocab = model["vocab"].as_array().ok_or_else(|| bad("Unigram model without vocab"))?;
        let mut pieces = HashMap::with_capacity(vocab.len());
        let (mut max_chars, mut min_score) = (1, f32::MAX);
        for (i, e) in vocab.iter().enumerate() {
            let (Some(p), Some(s)) = (e[0].as_str(), e[1].as_f64()) else { return Err(bad("malformed vocab entry")) };
            max_chars = max_chars.max(p.chars().count());
            min_score = min_score.min(s as f32);
            pieces.insert(p.to_string(), (i as u32, s as f32));
        }
        let unk = model["unk_id"].as_u64().ok_or_else(|| bad("Unigram model without unk_id"))? as u32;
        let id = |t: &str| pieces.get(t).map(|p| p.0).ok_or_else(|| bad(&format!("vocab lacks {t}")));
        let (cls, sep) = (id("<s>")?, id("</s>")?);
        let mut u = Unigram {
            unk,
            unk_score: min_score - UNK_PENALTY,
            cls,
            sep,
            pieces,
            max_chars,
            precompiled: None,
            collapse_spaces: false,
            lowercase: false,
            prefix_space: true,
        };
        let n = &tj["normalizer"];
        let steps: Vec<&Value> = match n["type"].as_str() {
            Some("Sequence") => n["normalizers"].as_array().map(|a| a.iter().collect()).unwrap_or_default(),
            Some(_) => vec![n],
            None => vec![],
        };
        for step in steps {
            match step["type"].as_str() {
                Some("Precompiled") => {
                    let b64 = step["precompiled_charsmap"].as_str().unwrap_or("");
                    if !b64.is_empty() {
                        let blob = crate::crypto::base64_decode(b64).ok_or_else(|| bad("charsmap is not base64"))?;
                        u.precompiled = Some(Precompiled::from_blob(&blob)?);
                    }
                }
                Some("Replace") if step["pattern"]["Regex"].as_str() == Some(" {2,}") => u.collapse_spaces = true,
                Some("Lowercase") => u.lowercase = true,
                Some(other) => return Err(bad(&format!("unsupported normalizer {other}"))),
                None => {}
            }
        }
        let pre = &tj["pre_tokenizer"];
        if pre["type"].as_str() != Some("Metaspace") {
            return Err(bad("Unigram tokenizers need a Metaspace pre-tokenizer"));
        }
        u.prefix_space =
            pre["add_prefix_space"].as_bool().unwrap_or(true) && pre["prepend_scheme"].as_str() != Some("never");
        Ok(u)
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Normalized text split into `▁word` pieces.
    pub fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut s = match &self.precompiled {
            Some(p) => p.normalize(text),
            None => text.to_string(),
        };
        if self.collapse_spaces {
            let mut out = String::with_capacity(s.len());
            for c in s.chars() {
                if !(c == ' ' && out.ends_with(' ')) {
                    out.push(c);
                }
            }
            s = out;
        }
        if self.lowercase {
            s = s.to_lowercase();
        }
        let mut s: String = s.chars().map(|c| if c == ' ' { SPACE } else { c }).collect();
        if self.prefix_space && !s.starts_with(SPACE) && !s.is_empty() {
            s.insert(0, SPACE);
        }
        let mut words: Vec<String> = Vec::new();
        for c in s.chars() {
            if c == SPACE || words.is_empty() {
                words.push(String::new());
            }
            words.last_mut().expect("pushed").push(c);
        }
        words
    }

    /// Best-scoring segmentation of one word (Viterbi), unknowns fused.
    fn segment(&self, word: &str, out: &mut Vec<u32>) {
        let starts: Vec<usize> = word.char_indices().map(|(i, _)| i).chain([word.len()]).collect();
        let n = starts.len() - 1;
        // best[i] = (score, start char index, id) of the best path ending at i.
        let mut best: Vec<(f32, usize, u32)> = vec![(f32::NEG_INFINITY, 0, 0); n + 1];
        best[0].0 = 0.0;
        for i in 0..n {
            if best[i].0 == f32::NEG_INFINITY {
                continue;
            }
            let mut single = false;
            for len in 1..=self.max_chars.min(n - i) {
                if let Some(&(id, score)) = self.pieces.get(&word[starts[i]..starts[i + len]]) {
                    single |= len == 1;
                    let s = best[i].0 + score;
                    if s > best[i + len].0 {
                        best[i + len] = (s, i, id);
                    }
                }
            }
            if !single {
                let s = best[i].0 + self.unk_score;
                if s > best[i + 1].0 {
                    best[i + 1] = (s, i, self.unk);
                }
            }
        }
        let mut path = Vec::new();
        let mut i = n;
        while i > 0 {
            let (_, start, id) = best[i];
            path.push(id);
            i = start;
        }
        path.reverse();
        path.dedup_by(|b, a| *a == self.unk && *b == self.unk);
        out.extend(path);
    }

    /// `<s> tokens… </s>`, truncated to `max_len` ids in total.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<u32> {
        let budget = max_len.max(2) - 2;
        let mut ids = vec![self.cls];
        let mut body = Vec::new();
        for w in self.pre_tokenize(text) {
            self.segment(&w, &mut body);
            if body.len() > budget {
                break;
            }
        }
        body.truncate(budget);
        ids.extend(body);
        ids.push(self.sep);
        ids
    }
}
