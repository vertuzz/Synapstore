//! OpenAI-compatible embeddings endpoint (`POST {base}/embeddings`), which
//! Ollama, llama.cpp, vLLM, LM Studio, TEI and OpenAI all speak.
//! `http://` uses a tiny built-in client; `https://` goes through `curl`.
//! The API key comes from `SYNAPSTORE_EMBED_API_KEY` (or `OPENAI_API_KEY`).

use super::Embedder;
use crate::error::{Error, Result};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub struct RemoteConfig {
    pub model: String,
    /// Base URL ending before `/embeddings`, e.g. `http://localhost:11434/v1`.
    pub base_url: String,
}

impl RemoteConfig {
    /// `model[@base_url]`; the default base URL is OpenAI's.
    pub fn parse(s: &str) -> Result<Self> {
        let (model, url) = s.split_once('@').unwrap_or((s, "https://api.openai.com/v1"));
        if model.trim().is_empty() || !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(Error::InvalidInput(format!("expected openai:<model>[@http(s)://host/v1], got `{s}`")));
        }
        Ok(RemoteConfig { model: model.trim().into(), base_url: url.trim_end_matches('/').into() })
    }
    pub fn label(&self) -> String {
        format!("{}@{}", self.model, self.base_url)
    }
}

pub struct RemoteEmbedder {
    cfg: RemoteConfig,
    source: String,
    dim: usize,
    key: Option<String>,
}

fn api_key() -> Option<String> {
    ["SYNAPSTORE_EMBED_API_KEY", "OPENAI_API_KEY"].iter().find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty()))
}

fn err(msg: impl Into<String>) -> Error {
    Error::InvalidInput(format!("embedding endpoint: {}", msg.into()))
}

/// Minimal HTTP/1.1 POST over plain TCP; returns (status, body).
pub fn http_post(url: &str, headers: &[(&str, String)], body: &[u8], timeout: Duration) -> Result<(u16, Vec<u8>)> {
    let rest = url.strip_prefix("http://").ok_or_else(|| err("not an http:// URL"))?;
    let (host, path) = rest.split_once('/').map_or((rest, "/".to_string()), |(h, p)| (h, format!("/{p}")));
    let addr = if host.contains(':') && !host.ends_with(']') { host.to_string() } else { format!("{host}:80") };
    let mut s = TcpStream::connect(&addr).map_err(|e| err(format!("connect {addr}: {e}")))?;
    s.set_read_timeout(Some(timeout))?;
    s.set_write_timeout(Some(timeout))?;
    let mut req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    s.write_all(req.as_bytes())?;
    s.write_all(body)?;
    let mut r = BufReader::new(s);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let status: u16 = line.split_whitespace().nth(1).and_then(|c| c.parse().ok()).ok_or_else(|| err("bad response"))?;
    let (mut len, mut chunked) = (None, false);
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
            if k == "content-length" {
                len = v.parse::<usize>().ok();
            } else if k == "transfer-encoding" && v.eq_ignore_ascii_case("chunked") {
                chunked = true;
            }
        }
    }
    let mut out = Vec::new();
    if chunked {
        loop {
            line.clear();
            r.read_line(&mut line)?;
            let n =
                usize::from_str_radix(line.trim().split(';').next().unwrap_or(""), 16).map_err(|_| err("bad chunk"))?;
            if n == 0 {
                break;
            }
            let start = out.len();
            out.resize(start + n, 0);
            r.read_exact(&mut out[start..])?;
            line.clear();
            r.read_line(&mut line)?;
        }
    } else if let Some(n) = len {
        out.resize(n, 0);
        r.read_exact(&mut out)?;
    } else {
        r.read_to_end(&mut out)?;
    }
    Ok((status, out))
}

fn curl_post(url: &str, key: Option<&str>, body: &str) -> Result<(u16, Vec<u8>)> {
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let mut cfg = format!("url = \"{}\"\nheader = \"Content-Type: application/json\"\n", esc(url));
    if let Some(k) = key {
        cfg.push_str(&format!("header = \"Authorization: Bearer {}\"\n", esc(k)));
    }
    cfg.push_str(&format!("data-binary = \"{}\"\n", esc(body)));
    let mut child = Command::new("curl")
        .args(["--silent", "--show-error", "--proto", "=https", "--max-time", "120", "--write-out", "\n%{http_code}"])
        .args(["--config", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| err(format!("cannot run curl: {e}")))?;
    child.stdin.take().expect("piped").write_all(cfg.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(err(String::from_utf8_lossy(&out.stderr).trim().to_string()));
    }
    let text = out.stdout;
    let split = text.iter().rposition(|&b| b == b'\n').unwrap_or(0);
    let code = String::from_utf8_lossy(&text[split..]).trim().parse().unwrap_or(0);
    Ok((code, text[..split].to_vec()))
}

impl RemoteEmbedder {
    /// Connect and probe the endpoint for its dimension.
    pub fn connect(cfg: RemoteConfig) -> Result<Self> {
        let mut e = RemoteEmbedder { source: String::new(), dim: 0, key: api_key(), cfg };
        let probe = e.request(&["dimension probe"])?;
        e.dim = probe[0].len();
        e.source = format!("remote:{}:{}", e.cfg.model, e.dim);
        Ok(e)
    }

    fn request(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.cfg.base_url);
        let body = json!({"model": self.cfg.model, "input": texts}).to_string();
        let (status, resp) = if url.starts_with("https://") {
            curl_post(&url, self.key.as_deref(), &body)?
        } else {
            let auth: Vec<(&str, String)> = self.key.iter().map(|k| ("Authorization", format!("Bearer {k}"))).collect();
            http_post(&url, &auth, body.as_bytes(), Duration::from_secs(120))?
        };
        let v: Value = serde_json::from_slice(&resp).map_err(|_| {
            err(format!("HTTP {status}: {}", String::from_utf8_lossy(&resp).chars().take(200).collect::<String>()))
        })?;
        if status != 200 {
            return Err(err(format!("HTTP {status}: {}", v["error"]["message"].as_str().unwrap_or(&v.to_string()))));
        }
        let mut data: Vec<&Value> = v["data"].as_array().ok_or_else(|| err("response lacks `data`"))?.iter().collect();
        data.sort_by_key(|d| d["index"].as_u64().unwrap_or(0));
        let out: Vec<Vec<f32>> = data
            .iter()
            .map(|d| d["embedding"].as_array().map(|a| a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect()))
            .collect::<Option<_>>()
            .ok_or_else(|| err("malformed embedding"))?;
        if out.len() != texts.len() || out.iter().any(|v| v.is_empty() || (self.dim > 0 && v.len() != self.dim)) {
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
        texts.chunks(64).map(|c| self.request(c)).collect::<Result<Vec<_>>>().map(|v| v.concat())
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.request(&[text])?.remove(0))
    }
}
