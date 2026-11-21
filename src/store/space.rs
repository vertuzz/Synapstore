//! Choosing the text embedder when a store is opened. The spec that built a
//! store's vectors is recorded in the file, so `--embedder` only has to be
//! given once; a conflicting choice is rejected before anything is written.

use super::Config;
use crate::embed::{Embedder, EmbedderSpec, SOURCE_BUILTIN, SOURCE_EXTERNAL};
use crate::error::{Error, Result};
use crate::logger::Log;
use crate::tables::{META, META_STR, keys, meta_get, meta_str_get};
use redb::{Database, ReadableDatabase, WriteTransaction};
use std::sync::Arc;

/// What the file says about its vector space.
#[derive(Debug, Default)]
pub struct Recorded {
    pub dim: Option<usize>,
    pub source: Option<String>,
    pub spec: Option<String>,
}

pub fn recorded(db: &Database) -> Result<Recorded> {
    let txn = db.begin_read()?;
    let (Ok(meta), Ok(ms)) = (txn.open_table(META), txn.open_table(META_STR)) else {
        return Ok(Recorded::default());
    };
    Ok(Recorded {
        dim: meta_get(&meta, keys::DIM)?.map(|d| d as usize),
        source: meta_str_get(&ms, keys::EMBEDDING_SOURCE)?,
        spec: meta_str_get(&ms, keys::EMBEDDER_SPEC)?,
    })
}

/// Build the embedder for this store (may download a model).
pub fn resolve(rec: &Recorded, cfg: &Config, log: &Log) -> Result<Arc<dyn Embedder>> {
    let spec = match (&cfg.embedder, &rec.spec) {
        (EmbedderSpec::Auto, Some(s)) => EmbedderSpec::parse(s)?,
        (s, _) => s.clone(),
    };
    let lexical_dim = match (rec.dim, rec.source.as_deref()) {
        (Some(d), Some(SOURCE_BUILTIN)) => d,
        _ => cfg.builtin_dim,
    };
    let progress = |m: &str| {
        log.info("embedder", &[("status", m.to_string())]);
        if cfg.embedder_progress {
            eprintln!("synapstore: {m}");
        }
    };
    let e = spec.build(lexical_dim, &progress)?;
    match rec.source.as_deref() {
        Some(src) if src != SOURCE_EXTERNAL && src != e.source() && !matches!(cfg.embedder, EmbedderSpec::Auto) => {
            let was = rec.spec.clone().unwrap_or_else(|| src.to_string());
            return Err(Error::InvalidInput(format!(
                "this store was built with embedder `{was}` (vectors `{src}`), not `{}`; \
                 reopen without --embedder or use a new file",
                e.source()
            )));
        }
        _ => {}
    }
    if let Some(d) = rec.dim
        && rec.source.as_deref() == Some(e.source())
        && d != e.dim()
    {
        return Err(Error::DimensionMismatch { expected: d, got: e.dim() });
    }
    log.info("embedder", &[("source", e.source().to_string()), ("dim", e.dim().to_string())]);
    Ok(e)
}

/// Remember the explicit spec so later opens pick the same embedder.
pub fn pin(txn: &WriteTransaction, rec: &Recorded, cfg: &Config, e: &dyn Embedder) -> Result<()> {
    let Some(label) = cfg.embedder.label().filter(|_| !matches!(cfg.embedder, EmbedderSpec::Custom(_))) else {
        return Ok(());
    };
    let compatible = rec.source.as_deref().is_none_or(|s| s == e.source());
    if compatible && rec.spec.as_deref() != Some(label.as_str()) {
        txn.open_table(META_STR)?.insert(keys::EMBEDDER_SPEC, label.as_str())?;
    }
    Ok(())
}
