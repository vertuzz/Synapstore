//! Built-in neural sentence embedder: BERT-family encoders (MiniLM, BGE,
//! …) loaded from Hugging Face `safetensors` + WordPiece vocabularies and
//! run on the CPU in pure Rust.

pub mod encoder;
pub mod precompiled;
pub mod safetensors;
pub mod tokenizer;
pub mod unicode;
pub mod unigram;

use super::Embedder;
use crate::crypto::{Sha256, hex};
use crate::error::{Error, Result};
use encoder::{Activation, BertConfig, Encoder};
use safetensors::SafeTensors;
use serde_json::Value;
use std::path::Path;
use tokenizer::{Normalization, WordPiece};
use unigram::Unigram;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pooling {
    Mean,
    Cls,
}

/// Options that do not live in the model files themselves.
#[derive(Debug, Clone, Default)]
pub struct BertOptions {
    /// Prepended to queries (e.g. BGE's retrieval instruction).
    pub query_prefix: String,
    /// Prepended to stored documents.
    pub doc_prefix: String,
    /// Override the pooling found in `1_Pooling/config.json`.
    pub pooling: Option<Pooling>,
    /// Override `sentence_bert_config.json`'s `max_seq_length`.
    pub max_tokens: Option<usize>,
}

/// The two tokenizer families of BERT-architecture sentence encoders.
pub enum Tokenizer {
    WordPiece(WordPiece),
    Unigram(Box<Unigram>),
}

impl Tokenizer {
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<u32> {
        match self {
            Tokenizer::WordPiece(t) => t.encode(text, max_len),
            Tokenizer::Unigram(t) => t.encode(text, max_len),
        }
    }
}

pub struct BertEmbedder {
    source: String,
    tokenizer: Tokenizer,
    encoder: Encoder,
    pooling: Pooling,
    max_tokens: usize,
    opts: BertOptions,
}

fn read_json(path: &Path) -> Result<Option<Value>> {
    match std::fs::read(path) {
        Ok(b) => {
            serde_json::from_slice(&b).map(Some).map_err(|e| Error::InvalidInput(format!("{}: {e}", path.display())))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn usize_field(v: &Value, key: &str) -> Result<usize> {
    v[key].as_u64().map(|x| x as usize).ok_or_else(|| Error::InvalidInput(format!("config.json lacks `{key}`")))
}

fn parse_config(v: &Value) -> Result<BertConfig> {
    let act = match v["hidden_act"].as_str().unwrap_or("gelu") {
        "gelu" => Activation::Gelu,
        "gelu_new" | "gelu_pytorch_tanh" | "gelu_fast" => Activation::GeluTanh,
        "relu" => Activation::Relu,
        other => return Err(Error::InvalidInput(format!("unsupported hidden_act `{other}`"))),
    };
    match v["model_type"].as_str() {
        Some("bert") | None => {}
        Some(other) => return Err(Error::InvalidInput(format!("unsupported model_type `{other}` (BERT only)"))),
    }
    Ok(BertConfig {
        vocab: usize_field(v, "vocab_size")?,
        hidden: usize_field(v, "hidden_size")?,
        layers: usize_field(v, "num_hidden_layers")?,
        heads: usize_field(v, "num_attention_heads")?,
        intermediate: usize_field(v, "intermediate_size")?,
        max_positions: usize_field(v, "max_position_embeddings")?,
        type_vocab: v["type_vocab_size"].as_u64().unwrap_or(2) as usize,
        eps: v["layer_norm_eps"].as_f64().unwrap_or(1e-12) as f32,
        act,
    })
}

/// Tokenizer from `tokenizer.json` (preferred) or `vocab.txt` +
/// `tokenizer_config.json`.
fn load_tokenizer(dir: &Path) -> Result<(Tokenizer, Vec<u8>)> {
    if let Some(tj) = read_json(&dir.join("tokenizer.json"))? {
        let model = &tj["model"];
        let bytes = std::fs::read(dir.join("tokenizer.json"))?;
        match model["type"].as_str() {
            Some("WordPiece") => {}
            Some("Unigram") => return Ok((Tokenizer::Unigram(Box::new(Unigram::from_json(&tj)?)), bytes)),
            other => {
                return Err(Error::InvalidInput(format!("tokenizer.json: unsupported model {other:?}")));
            }
        }
        let n = &tj["normalizer"];
        let lowercase = n["lowercase"].as_bool().unwrap_or(true);
        let norm = Normalization {
            lowercase,
            strip_accents: n["strip_accents"].as_bool().unwrap_or(lowercase),
            chinese_chars: n["handle_chinese_chars"].as_bool().unwrap_or(true),
        };
        let vocab = model["vocab"].as_object().ok_or_else(|| Error::InvalidInput("tokenizer.json: no vocab".into()))?;
        let map = vocab.iter().filter_map(|(k, v)| Some((k.clone(), v.as_u64()? as u32))).collect();
        return Ok((Tokenizer::WordPiece(WordPiece::from_map(map, norm)?), bytes));
    }
    let vocab = std::fs::read_to_string(dir.join("vocab.txt"))
        .map_err(|e| Error::InvalidInput(format!("{}: no tokenizer.json or vocab.txt ({e})", dir.display())))?;
    let tc = read_json(&dir.join("tokenizer_config.json"))?.unwrap_or(Value::Null);
    let lowercase = tc["do_lower_case"].as_bool().unwrap_or(true);
    let norm = Normalization {
        lowercase,
        strip_accents: tc["strip_accents"].as_bool().unwrap_or(lowercase),
        chinese_chars: tc["tokenize_chinese_chars"].as_bool().unwrap_or(true),
    };
    let bytes = vocab.as_bytes().to_vec();
    Ok((Tokenizer::WordPiece(WordPiece::from_vocab(&vocab, norm)?), bytes))
}

impl BertEmbedder {
    /// Load a model directory laid out like a Hugging Face repository.
    pub fn load(dir: &Path, name: &str, opts: BertOptions) -> Result<Self> {
        let cfg_json = read_json(&dir.join("config.json"))?
            .ok_or_else(|| Error::InvalidInput(format!("{}: missing config.json", dir.display())))?;
        let cfg = parse_config(&cfg_json)?;
        let (tokenizer, tok_bytes) = load_tokenizer(dir)?;
        let weights = dir.join("model.safetensors");
        let mut st = SafeTensors::open(&weights)?;
        let encoder = Encoder::load(&mut st, cfg)?;
        let pool_cfg = read_json(&dir.join("1_Pooling/config.json"))?.unwrap_or(Value::Null);
        let pooling = opts.pooling.unwrap_or(if pool_cfg["pooling_mode_cls_token"].as_bool() == Some(true) {
            Pooling::Cls
        } else {
            Pooling::Mean
        });
        let sbert = read_json(&dir.join("sentence_bert_config.json"))?.unwrap_or(Value::Null);
        let max_tokens = opts
            .max_tokens
            .or(sbert["max_seq_length"].as_u64().map(|m| m as usize))
            .unwrap_or(512)
            .clamp(8, cfg.max_positions);
        // Fingerprint the exact weights + vocabulary cheaply (header, size,
        // tokenizer) so a store notices if the model behind a name changes.
        let mut h = Sha256::new();
        let mut head = std::fs::metadata(&weights)?.len().to_le_bytes().to_vec();
        head.extend(st.names().count().to_le_bytes());
        h.update(&head);
        h.update(&safetensors_header(&weights)?);
        h.update(&tok_bytes);
        h.update(format!("{pooling:?}|{}|{}", opts.query_prefix, opts.doc_prefix).as_bytes());
        let fp = hex(&h.finish()[..6]);
        Ok(BertEmbedder { source: format!("bert:{name}:{fp}"), tokenizer, encoder, pooling, max_tokens, opts })
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Token ids for `text` as the model sees them.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text, self.max_tokens)
    }

    /// Pooled, L2-normalized embedding of exactly `text` (no prefix).
    pub fn embed_raw(&self, text: &str) -> Vec<f32> {
        self.embed_batch(&[text]).remove(0)
    }

    /// Embed several texts in one forward pass.
    pub fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        let ids: Vec<Vec<u32>> = texts.iter().map(|t| self.tokenize(t)).collect();
        let h = self.encoder.cfg.hidden;
        self.encoder
            .forward(&ids)
            .into_iter()
            .map(|hs| {
                let v: Vec<f32> = match self.pooling {
                    Pooling::Cls => hs[..h].to_vec(),
                    Pooling::Mean => {
                        let n = (hs.len() / h).max(1);
                        (0..h).map(|i| (0..n).map(|t| hs[t * h + i]).sum::<f32>() / n as f32).collect()
                    }
                };
                crate::math::normalize(&v).unwrap_or_else(|_| {
                    let mut e = vec![0.0; h];
                    e[0] = 1.0;
                    e
                })
            })
            .collect()
    }
}

fn safetensors_header(path: &Path) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut buf = vec![0u8; (u64::from_le_bytes(len) as usize).min(100 << 20)];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

impl Embedder for BertEmbedder {
    fn source(&self) -> &str {
        &self.source
    }
    fn dim(&self) -> usize {
        self.encoder.cfg.hidden
    }
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let docs: Vec<String> = texts.iter().map(|t| format!("{}{t}", self.opts.doc_prefix)).collect();
        // Batches of up to 8 texts, spread over the available cores.
        let groups: Vec<Vec<&str>> = docs.chunks(8).map(|g| g.iter().map(String::as_str).collect()).collect();
        Ok(super::parallel_map(&groups, |g| self.embed_batch(g)).concat())
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed_raw(&format!("{}{text}", self.opts.query_prefix)))
    }
    fn chunk_chars(&self) -> Option<usize> {
        // ~3.5 characters per WordPiece token for English prose.
        Some((self.max_tokens.saturating_sub(2) * 7 / 2).max(200))
    }
}
