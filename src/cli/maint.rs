//! Maintenance commands: `reindex` (migrate to another embedder) and
//! `eval` (recall regression check).

use super::Cli;
use super::run::{json, open};
use crate::SynapStore;
use std::path::Path;

pub fn reindex(c: &Cli, src: &Path, dst: &Path, merge: bool) -> Result<(), String> {
    let same = std::fs::canonicalize(src).ok().zip(std::fs::canonicalize(dst).ok()).is_some_and(|(a, b)| a == b);
    if same {
        return Err("SRC and DST must be different files".into());
    }
    if !src.is_file() {
        return Err(format!("{} does not exist", src.display()));
    }
    if dst.exists() {
        return Err(format!("{} already exists; reindex writes a new file", dst.display()));
    }
    let mut cfg = c.config.clone();
    cfg.embedder_progress = true;
    let store = SynapStore::open(dst, cfg).map_err(|e| format!("cannot create {}: {e}", dst.display()))?;
    let progress = |m: &str| eprintln!("synapstore: {m}");
    let rep = crate::reindex::reindex(src, &store, merge, &progress).map_err(|e| e.to_string());
    drop(store);
    match rep {
        Ok(r) => json(&r),
        Err(e) => {
            std::fs::remove_file(dst).ok();
            Err(format!("reindex failed (removed the partial {}): {e}", dst.display()))
        }
    }
}

pub fn eval(c: &Cli, queries: &Path, top_k: usize) -> Result<(), String> {
    let text = std::fs::read_to_string(queries).map_err(|e| format!("cannot read {}: {e}", queries.display()))?;
    let cases = crate::eval::parse_cases(&text).map_err(|e| e.to_string())?;
    let store = open(c)?;
    json(&crate::eval::run(&store, &cases, top_k).map_err(|e| e.to_string())?)
}
