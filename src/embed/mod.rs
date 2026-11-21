//! Text embedders. A store is pinned to one embedding space by its first
//! write; the embedder that produced it is remembered in the file, so later
//! opens pick the same one automatically.
//!
//! * [`lexical`] — dependency-free feature hashing (default, no model).
//! * [`bert`] — neural sentence embeddings (MiniLM, BGE, …) run on the CPU
//!   in pure Rust; weights are fetched once by [`fetch`].
//! * [`remote`] — any OpenAI-compatible `/v1/embeddings` endpoint
//!   (Ollama, llama.cpp, vLLM, TEI, OpenAI, …).

pub mod bert;
pub mod curl;
pub mod fetch;
pub mod lexical;
pub mod remote;

pub use lexical::{BUILTIN_DIM, SOURCE_BUILTIN, SOURCE_EXTERNAL, embed, stem, tokens};

use crate::error::{Error, Result};
use std::sync::Arc;

/// Something that turns text into vectors of a fixed dimension.
pub trait Embedder: Send + Sync {
    /// Identifier of the vector space, persisted in the store.
    fn source(&self) -> &str;
    fn dim(&self) -> usize;
    /// Embed texts that will be stored.
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    /// Embed a search query (some models use a query-specific prefix).
    fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
    /// Preferred chunk size in characters (models have a token limit).
    fn chunk_chars(&self) -> Option<usize> {
        None
    }
}

/// The built-in lexical embedder as an [`Embedder`].
#[derive(Debug, Clone, Copy)]
pub struct Lexical {
    pub dim: usize,
}

impl Embedder for Lexical {
    fn source(&self) -> &str {
        SOURCE_BUILTIN
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| embed(t, self.dim)).collect())
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(embed(text, self.dim))
    }
}

/// Map `f` over `items` on up to `available_parallelism` threads, in order.
pub fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(items.len());
    if threads <= 1 {
        return items.iter().map(f).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<(usize, R)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(item) = items.get(i) else { break };
                        local.push((i, f(item)));
                    }
                    local
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("embedding worker panicked")).collect()
    });
    out.sort_by_key(|(i, _)| *i);
    out.into_iter().map(|(_, r)| r).collect()
}

/// Which embedder a store should use for text.
#[derive(Clone, Default)]
pub enum EmbedderSpec {
    /// Whatever the store was created with; lexical for a new store.
    #[default]
    Auto,
    Lexical,
    /// A model from the [`fetch::REGISTRY`] (`minilm`, `bge-small`, …),
    /// `name@revision`, `hf:owner/repo[@revision]`, or a local directory.
    Model(String),
    Remote(remote::RemoteConfig),
    /// A caller-supplied implementation (library use).
    Custom(Arc<dyn Embedder>),
}

impl std::fmt::Debug for EmbedderSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.label() {
            Some(l) => write!(f, "EmbedderSpec({l})"),
            None => write!(f, "EmbedderSpec(auto)"),
        }
    }
}

impl EmbedderSpec {
    /// Parse a `--embedder` value.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        Ok(match s {
            "" | "auto" => EmbedderSpec::Auto,
            "lexical" | "builtin" | "hash" => EmbedderSpec::Lexical,
            _ if s.starts_with("openai:") => EmbedderSpec::Remote(remote::RemoteConfig::parse(&s[7..])?),
            _ => EmbedderSpec::Model(s.to_string()),
        })
    }

    /// The string persisted in the store (`None` for auto/custom).
    pub fn label(&self) -> Option<String> {
        match self {
            EmbedderSpec::Auto => None,
            EmbedderSpec::Lexical => Some("lexical".into()),
            EmbedderSpec::Model(m) => Some(m.clone()),
            EmbedderSpec::Remote(r) => Some(format!("openai:{}", r.label())),
            EmbedderSpec::Custom(e) => Some(format!("custom:{}", e.source())),
        }
    }

    /// Instantiate. `lexical_dim` sizes the lexical embedder; `progress`
    /// receives human-readable status lines (model downloads).
    pub fn build(&self, lexical_dim: usize, progress: &dyn Fn(&str)) -> Result<Arc<dyn Embedder>> {
        Ok(match self {
            EmbedderSpec::Auto | EmbedderSpec::Lexical => Arc::new(Lexical { dim: lexical_dim }),
            EmbedderSpec::Model(name) => Arc::new(fetch::load_model(name, progress)?),
            EmbedderSpec::Remote(cfg) => Arc::new(remote::RemoteEmbedder::connect(cfg.clone())?),
            EmbedderSpec::Custom(e) => e.clone(),
        })
    }
}

/// Embed one query and check it lands in a store of dimension `dim`.
pub fn query_in_space(e: &dyn Embedder, text: &str, dim: usize) -> Result<Vec<f32>> {
    let v = crate::math::normalize(&e.embed_query(text)?)?;
    if v.len() != dim {
        return Err(Error::DimensionMismatch { expected: dim, got: v.len() });
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_map_keeps_order() {
        let v: Vec<usize> = (0..257).collect();
        assert_eq!(parallel_map(&v, |x| x * 2), v.iter().map(|x| x * 2).collect::<Vec<_>>());
        assert!(parallel_map(&Vec::<u8>::new(), |x| *x).is_empty());
    }

    #[test]
    fn parses_specs() {
        assert!(matches!(EmbedderSpec::parse("auto").unwrap(), EmbedderSpec::Auto));
        assert!(matches!(EmbedderSpec::parse("lexical").unwrap(), EmbedderSpec::Lexical));
        assert_eq!(EmbedderSpec::parse("minilm").unwrap().label().unwrap(), "minilm");
        let r = EmbedderSpec::parse("openai:nomic-embed-text@http://localhost:11434/v1").unwrap();
        assert_eq!(r.label().unwrap(), "openai:nomic-embed-text@http://localhost:11434/v1");
    }
}
