//! Model registry and one-time download from the Hugging Face Hub.
//!
//! Downloads shell out to `curl` (bundled with macOS, Windows 10+ and every
//! Linux distribution) so no TLS stack enters the dependency budget. Large
//! files are verified against the SHA-256 the Hub publishes in
//! `X-Linked-Etag`; a model directory is only used once complete.
//!
//! Environment: `SYNAPSTORE_MODELS` (cache dir), `HF_ENDPOINT` (mirror),
//! `HF_TOKEN` (gated/private repos), `HF_HUB_OFFLINE=1` (never download).

use super::bert::{BertEmbedder, BertOptions, Pooling};
use super::curl::fetch;
use crate::error::{Error, Result};
use std::path::{Path, PathBuf};

/// A curated model that `--embedder <name>` can fetch.
#[derive(Debug, Clone, Copy)]
pub struct ModelInfo {
    pub name: &'static str,
    pub repo: &'static str,
    pub dim: usize,
    pub download_mb: u32,
    pub license: &'static str,
    pub languages: &'static str,
    pub query_prefix: &'static str,
    pub doc_prefix: &'static str,
    pub pooling: Option<Pooling>,
    pub about: &'static str,
}

pub const REGISTRY: &[ModelInfo] = &[
    ModelInfo {
        name: "minilm",
        repo: "sentence-transformers/all-MiniLM-L6-v2",
        dim: 384,
        download_mb: 91,
        license: "Apache-2.0",
        languages: "en",
        query_prefix: "",
        doc_prefix: "",
        pooling: Some(Pooling::Mean),
        about: "fast general-purpose sentence embeddings (6 layers)",
    },
    ModelInfo {
        name: "bge-small",
        repo: "BAAI/bge-small-en-v1.5",
        dim: 384,
        download_mb: 134,
        license: "MIT",
        languages: "en",
        query_prefix: "Represent this sentence for searching relevant passages: ",
        doc_prefix: "",
        pooling: Some(Pooling::Cls),
        about: "stronger English retrieval (12 layers, ~2x slower)",
    },
    ModelInfo {
        name: "multilingual",
        repo: "intfloat/multilingual-e5-small",
        dim: 384,
        download_mb: 471,
        license: "MIT",
        languages: "~100 languages: ru, uk, de, fr, es, zh, ja, ar, …",
        query_prefix: "query: ",
        doc_prefix: "passage: ",
        pooling: Some(Pooling::Mean),
        about: "cross-lingual retrieval (12 layers, ~2x slower, ~500 MB RAM)",
    },
];

const REQUIRED: &[&str] = &["config.json", "model.safetensors"];
const OPTIONAL: &[&str] =
    &["tokenizer.json", "vocab.txt", "tokenizer_config.json", "1_Pooling/config.json", "sentence_bert_config.json"];
const COMPLETE: &str = ".synapstore-complete";

/// Root of the model cache.
pub fn models_dir() -> PathBuf {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(d) = env("SYNAPSTORE_MODELS") {
        return d;
    }
    let base = env("XDG_CACHE_HOME")
        .or_else(|| env("LOCALAPPDATA"))
        .or_else(|| env("HOME").map(|h| h.join(".cache")))
        .or_else(|| env("USERPROFILE").map(|h| h.join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("synapstore").join("models")
}

/// What a `--embedder` model string refers to.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRef {
    pub repo: String,
    pub revision: String,
    pub info: Option<ModelInfo>,
}

impl PartialEq for ModelInfo {
    fn eq(&self, o: &Self) -> bool {
        self.name == o.name
    }
}

/// Resolve `minilm`, `minilm@<rev>` or `hf:owner/repo[@rev]`.
pub fn resolve(spec: &str) -> Result<ModelRef> {
    let (base, revision) = spec.split_once('@').unwrap_or((spec, "main"));
    let ok = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "-_./".contains(c));
    if !ok(revision) || revision.contains("..") {
        return Err(Error::InvalidInput(format!("bad model revision `{revision}`")));
    }
    if let Some(repo) = base.strip_prefix("hf:") {
        if !ok(repo) || repo.matches('/').count() != 1 || repo.contains("..") {
            return Err(Error::InvalidInput(format!("bad Hugging Face repo `{repo}` (expected owner/name)")));
        }
        let info = REGISTRY.iter().find(|m| m.repo == repo).copied();
        return Ok(ModelRef { repo: repo.into(), revision: revision.into(), info });
    }
    match REGISTRY.iter().find(|m| m.name == base) {
        Some(m) => Ok(ModelRef { repo: m.repo.into(), revision: revision.into(), info: Some(*m) }),
        None => {
            let names: Vec<_> = REGISTRY.iter().map(|m| m.name).collect();
            Err(Error::InvalidInput(format!(
                "unknown embedder `{base}`: use lexical, {}, hf:<owner>/<repo>, openai:<model>@<url> or a model directory",
                names.join(", ")
            )))
        }
    }
}

impl ModelRef {
    pub fn cache_dir(&self) -> PathBuf {
        models_dir().join(self.repo.replace('/', "--")).join(self.revision.replace('/', "--"))
    }
    fn options(&self) -> BertOptions {
        let i = self.info;
        BertOptions {
            query_prefix: i.map(|m| m.query_prefix.to_string()).unwrap_or_default(),
            doc_prefix: i.map(|m| m.doc_prefix.to_string()).unwrap_or_default(),
            pooling: i.and_then(|m| m.pooling),
            ..BertOptions::default()
        }
    }
    fn name(&self) -> String {
        let base = self.info.map(|m| m.name.to_string()).unwrap_or_else(|| self.repo.clone());
        if self.revision == "main" { base } else { format!("{base}@{}", self.revision) }
    }
}

/// Load (downloading on first use) the model named by `spec`, or a local
/// model directory.
pub fn load_model(spec: &str, progress: &dyn Fn(&str)) -> Result<BertEmbedder> {
    let path = Path::new(spec);
    if path.join("config.json").is_file() {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "local".into());
        return BertEmbedder::load(path, &name, BertOptions::default());
    }
    let r = resolve(spec)?;
    let dir = r.cache_dir();
    if !dir.join(COMPLETE).is_file() {
        download(&r, &dir, progress)?;
    }
    let start = std::time::Instant::now();
    let e = BertEmbedder::load(&dir, &r.name(), r.options())?;
    progress(&format!("loaded {} in {} ms", r.name(), start.elapsed().as_millis()));
    Ok(e)
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "https://huggingface.co".into())
}

/// Fetch every model file of `r` into `dir` (atomically).
pub fn download(r: &ModelRef, dir: &Path, progress: &dyn Fn(&str)) -> Result<()> {
    if std::env::var("HF_HUB_OFFLINE").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true")) {
        return Err(Error::InvalidInput(format!("model {} is not cached and HF_HUB_OFFLINE is set", r.repo)));
    }
    let size = r.info.map(|m| format!(" (~{} MB)", m.download_mb)).unwrap_or_default();
    progress(&format!("downloading {}@{}{size} into {}", r.repo, r.revision, dir.display()));
    let parent = dir.parent().ok_or_else(|| Error::InvalidInput("bad cache dir".into()))?;
    std::fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(".partial-{}-{}", std::process::id(), crate::timefmt::now_ms()));
    let result = (|| {
        let mut manifest = String::new();
        for (i, file) in REQUIRED.iter().chain(OPTIONAL).enumerate() {
            let url = format!("{}/{}/resolve/{}/{file}", endpoint(), r.repo, r.revision);
            let dest = tmp.join(file);
            std::fs::create_dir_all(dest.parent().unwrap_or(&tmp))?;
            match fetch(&url, &dest, file.ends_with(".safetensors"))? {
                Some(sha) => manifest.push_str(&format!("{sha}  {file}\n")),
                None if i < REQUIRED.len() => {
                    return Err(Error::InvalidInput(format!("{url} not found (is it a BERT model with safetensors?)")));
                }
                None => {}
            }
        }
        if !tmp.join("tokenizer.json").is_file() && !tmp.join("vocab.txt").is_file() {
            return Err(Error::InvalidInput(format!("{} has neither tokenizer.json nor vocab.txt", r.repo)));
        }
        // Validate before publishing the directory.
        BertEmbedder::load(&tmp, &r.repo, r.options())?;
        std::fs::write(tmp.join(COMPLETE), manifest)?;
        if dir.exists() {
            std::fs::remove_dir_all(dir)?;
        }
        std::fs::rename(&tmp, dir)?;
        Ok(())
    })();
    if result.is_err() {
        std::fs::remove_dir_all(&tmp).ok();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_names() {
        let r = resolve("minilm").unwrap();
        assert_eq!((r.repo.as_str(), r.revision.as_str()), ("sentence-transformers/all-MiniLM-L6-v2", "main"));
        assert_eq!(resolve("bge-small@v1").unwrap().name(), "bge-small@v1");
        assert_eq!(resolve("hf:a/b").unwrap().repo, "a/b");
        assert!(resolve("hf:../x").is_err() && resolve("hf:a").is_err() && resolve("nope").is_err());
        assert!(resolve("minilm@../../etc").is_err());
    }
}
