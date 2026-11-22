//! OpenAI-compatible embeddings endpoint (`POST {base}/embeddings`), which
//! Ollama, llama.cpp, vLLM, LM Studio, TEI, OpenAI and OpenRouter all speak.
//!
//! * `openai:<model>[@<base>]` — key from `SYNAPSTORE_EMBED_API_KEY` or `OPENAI_API_KEY`.
//! * `openrouter:<model>[@<base>]` — key from `OPENROUTER_API_KEY` (or
//!   `SYNAPSTORE_EMBED_API_KEY`); every request asks OpenRouter to route only
//!   to zero-data-retention providers that do not train on the data.
//!
//! Transient failures (network, HTTP 408/429/5xx) are retried with backoff.
//! When the store already records the dimension the endpoint is not probed
//! at startup, so a server starts (and `stats` works) without network.

use super::Embedder;
use super::http::{err, post_json_timeout};
use crate::error::Result;
use serde_json::{Value, json};
use std::time::Duration;

const OPENAI_URL: &str = "https://api.openai.com/v1";
const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1";
/// Chunk size for remote models: their context windows hold whole memories.
pub const REMOTE_CHUNK_CHARS: usize = 8000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Provider {
    #[default]
    OpenAi,
    OpenRouter,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RemoteConfig {
    pub model: String,
    /// Base URL ending before `/embeddings`, e.g. `http://localhost:11434/v1`.
    pub base_url: String,
    pub provider: Provider,
}

impl RemoteConfig {
    /// `model[@base_url]`; the default base URL is OpenAI's.
    pub fn parse(s: &str) -> Result<Self> {
        Self::parse_for(s, Provider::OpenAi)
    }

    /// `model[@base_url]` for `provider`.
    pub fn parse_for(s: &str, provider: Provider) -> Result<Self> {
        let default = match provider {
            Provider::OpenAi => OPENAI_URL,
            Provider::OpenRouter => OPENROUTER_URL,
        };
        let (model, url) = s.split_once('@').unwrap_or((s, default));
        if model.trim().is_empty() || !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(crate::Error::InvalidInput(format!(
                "expected {}:<model>[@http(s)://host/v1], got `{s}`",
                Self::prefix_of(provider)
            )));
        }
        Ok(RemoteConfig { model: model.trim().into(), base_url: url.trim_end_matches('/').into(), provider })
    }

    fn prefix_of(p: Provider) -> &'static str {
        match p {
            Provider::OpenAi => "openai",
            Provider::OpenRouter => "openrouter",
        }
    }

    /// `model@base_url`.
    pub fn label(&self) -> String {
        format!("{}@{}", self.model, self.base_url)
    }

    /// The full `--embedder` spec (persisted in the store).
    pub fn spec(&self) -> String {
        let p = Self::prefix_of(self.provider);
        match self.provider {
            Provider::OpenRouter if self.base_url == OPENROUTER_URL => format!("{p}:{}", self.model),
            _ => format!("{p}:{}", self.label()),
        }
    }

    /// The vector-space id for this model at `dim`.
    pub fn source(&self, dim: usize) -> String {
        format!("remote:{}:{dim}", self.model)
    }

    fn key_vars(&self) -> &'static [&'static str] {
        match self.provider {
            Provider::OpenAi => &["SYNAPSTORE_EMBED_API_KEY", "OPENAI_API_KEY"],
            Provider::OpenRouter => &["OPENROUTER_API_KEY", "SYNAPSTORE_EMBED_API_KEY"],
        }
    }

    /// Prefixes some model families expect for (query, document) texts.
    fn prefixes(&self) -> (&'static str, &'static str) {
        let m = self.model.to_ascii_lowercase();
        let name = m.rsplit('/').next().unwrap_or(&m);
        if name.starts_with("e5-") || name.starts_with("multilingual-e5") {
            ("query: ", "passage: ")
        } else if name.starts_with("qwen3-embedding") {
            ("Instruct: Given a question, retrieve memory notes that answer it\nQuery: ", "")
        } else {
            ("", "")
        }
    }
}

/// Retry schedule for one kind of request.
#[derive(Debug, Clone, Copy)]
struct Attempts {
    tries: u32,
    timeout: Duration,
    backoff: Duration,
}

const DOCS: Attempts = Attempts { tries: 4, timeout: Duration::from_secs(120), backoff: Duration::from_secs(1) };
const QUERY: Attempts = Attempts { tries: 3, timeout: Duration::from_secs(20), backoff: Duration::from_millis(400) };

pub struct RemoteEmbedder {
    cfg: RemoteConfig,
    source: String,
    dim: usize,
    key: Option<String>,
}

impl RemoteEmbedder {
    /// Connect and probe the endpoint for its dimension.
    pub fn connect(cfg: RemoteConfig) -> Result<Self> {
        Self::with_dim(cfg, None)
    }

    /// Use a known dimension (no network at startup), or probe when `None`.
    pub fn with_dim(cfg: RemoteConfig, dim: Option<usize>) -> Result<Self> {
        let key = cfg.key_vars().iter().find_map(|k| std::env::var(k).ok().filter(|v| !v.trim().is_empty()));
        let mut e = RemoteEmbedder { source: String::new(), dim: 0, key, cfg };
        e.dim = match dim {
            Some(d) if d > 0 => d,
            _ => e.request(&["dimension probe"], QUERY)?[0].len(),
        };
        e.source = e.cfg.source(e.dim);
        Ok(e)
    }

    fn body(&self, texts: &[&str]) -> String {
        let mut b = json!({"model": self.cfg.model, "input": texts});
        if self.cfg.provider == Provider::OpenRouter {
            b["provider"] = json!({"zdr": true, "data_collection": "deny"});
        }
        b.to_string()
    }

    fn request(&self, texts: &[&str], a: Attempts) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.cfg.base_url);
        let body = self.body(texts);
        let mut wait = a.backoff;
        let mut last = err("no attempt made");
        for attempt in 1..=a.tries {
            match post_json_timeout(&url, self.key.as_deref(), &body, a.timeout) {
                Ok((200, resp)) => return self.parse(texts.len(), 200, &resp),
                Ok((status, resp)) => {
                    last =
                        self.parse(texts.len(), status, &resp).err().unwrap_or_else(|| err(format!("HTTP {status}")));
                    if !(status == 408 || status == 429 || status >= 500) {
                        return Err(last);
                    }
                }
                Err(e) => last = e,
            }
            if attempt < a.tries {
                std::thread::sleep(wait);
                wait *= 2;
            }
        }
        Err(last)
    }

    fn parse(&self, n: usize, status: u16, resp: &[u8]) -> Result<Vec<Vec<f32>>> {
        let v: Value = serde_json::from_slice(resp).map_err(|_| {
            err(format!("HTTP {status}: {}", String::from_utf8_lossy(resp).chars().take(200).collect::<String>()))
        })?;
        if status != 200 {
            return Err(err(format!("HTTP {status}: {}", v["error"]["message"].as_str().unwrap_or(&v.to_string()))));
        }
        let Some(arr) = v["data"].as_array() else {
            let msg =
                v["error"]["message"].as_str().map_or_else(|| "response lacks `data`".to_string(), str::to_string);
            return Err(err(msg));
        };
        let mut data: Vec<&Value> = arr.iter().collect();
        data.sort_by_key(|d| d["index"].as_u64().unwrap_or(0));
        let out: Vec<Vec<f32>> = data
            .iter()
            .map(|d| d["embedding"].as_array().map(|a| a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect()))
            .collect::<Option<_>>()
            .ok_or_else(|| err("malformed embedding"))?;
        if out.len() != n || out.iter().any(|v| v.is_empty() || (self.dim > 0 && v.len() != self.dim)) {
            return Err(err("wrong number or size of embeddings"));
        }
        Ok(out)
    }
}

impl Embedder for RemoteEmbedder {
    fn source(&self) -> &str {
        &self.source
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let pre = self.cfg.prefixes().1;
        let docs: Vec<String> = texts.iter().map(|t| format!("{pre}{t}")).collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        refs.chunks(32).map(|c| self.request(c, DOCS)).collect::<Result<Vec<_>>>().map(|v| v.concat())
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let q = format!("{}{text}", self.cfg.prefixes().0);
        Ok(self.request(&[&q], QUERY)?.remove(0))
    }
    fn chunk_chars(&self) -> Option<usize> {
        Some(REMOTE_CHUNK_CHARS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_round_trip() {
        let r = RemoteConfig::parse_for("perplexity/pplx-embed-v1-4b", Provider::OpenRouter).unwrap();
        assert_eq!(r.base_url, OPENROUTER_URL);
        assert_eq!(r.spec(), "openrouter:perplexity/pplx-embed-v1-4b");
        assert_eq!(r.source(2560), "remote:perplexity/pplx-embed-v1-4b:2560");
        let o = RemoteConfig::parse("m@http://localhost:1/v1/").unwrap();
        assert_eq!(o.spec(), "openai:m@http://localhost:1/v1");
        assert!(RemoteConfig::parse_for("@http://x", Provider::OpenRouter).is_err());
        assert!(RemoteConfig::parse("m@ftp://x").is_err());
    }

    #[test]
    fn prefixes_by_family() {
        let p = |m: &str| RemoteConfig::parse(m).unwrap().prefixes();
        assert_eq!(p("intfloat/multilingual-e5-large").0, "query: ");
        assert_eq!(p("e5-small").1, "passage: ");
        assert!(p("qwen/qwen3-embedding-8b").0.starts_with("Instruct:"));
        assert_eq!(p("perplexity/pplx-embed-v1-4b"), ("", ""));
    }

    #[test]
    fn openrouter_body_requests_zero_retention() {
        let cfg = RemoteConfig::parse_for("m@http://127.0.0.1:1/v1", Provider::OpenRouter).unwrap();
        let e = RemoteEmbedder { cfg, source: String::new(), dim: 2, key: None };
        let b: Value = serde_json::from_str(&e.body(&["x"])).unwrap();
        assert_eq!(b["provider"]["zdr"], true);
        assert_eq!(b["provider"]["data_collection"], "deny");
        let cfg = RemoteConfig::parse("m@http://127.0.0.1:1/v1").unwrap();
        let e = RemoteEmbedder { cfg, source: String::new(), dim: 2, key: None };
        assert!(serde_json::from_str::<Value>(&e.body(&["x"])).unwrap().get("provider").is_none());
    }
}
