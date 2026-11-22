//! Telemetry and whitebox export (PRD §9): `stats` for agents tuning
//! hyper-parameters, `dump` for a full JSON-lines export of every table.

use super::SynapStore;
use crate::archive;
use crate::error::{Error, Result};
use crate::hash::{IdMap, id_to_hex};
use crate::hnsw::{EntryPoint, inspect};
use crate::model::{Chunk, HnswNode, MemoryItem, Term, Triple};
use crate::tables::*;
use redb::{ReadableDatabase, ReadableTable, ReadableTableMetadata};
use serde_json::{Value, json};
use std::io::Write;

fn ser(e: impl std::fmt::Display) -> Error {
    Error::Serialization(e.to_string())
}

impl SynapStore {
    /// Sizes, index structure, configuration and live metrics.
    /// `detailed` scans the whole HNSW graph (O(n)).
    pub fn stats(&self, detailed: bool) -> Result<Value> {
        let txn = self.db().begin_read()?;
        let meta = txn.open_table(META)?;
        let chunks = txn.open_table(CHUNKS)?.len()?;
        let nodes_t = txn.open_table(HNSW_NODES)?;
        let nodes = nodes_t.len()?;
        let ep = EntryPoint::load(&meta)?;
        let cfg = self.config();
        let mut index = json!({
            "nodes": nodes,
            "tombstones": nodes.saturating_sub(chunks),
            "entry_point": ep.map(|e| id_to_hex(e.id)),
            "max_level": ep.map(|e| e.max_level),
            "m": cfg.hnsw.m,
            "m0": cfg.hnsw.m0,
            "ef_construction": cfg.hnsw.ef_construction,
            "ef_search": cfg.hnsw.ef_search,
        });
        if detailed {
            let st = inspect::index_stats(&nodes_t)?;
            index["level_histogram"] = json!(st.level_histogram);
            index["mean_degree"] = json!(st.mean_degree);
            index["isolated_nodes"] = json!(st.isolated);
            index["reachable_from_entry"] = json!(inspect::reachable_from_entry(&nodes_t, ep)?);
        }
        let file_bytes = std::fs::metadata(self.path()).map(|m| m.len()).unwrap_or(0);
        let d = cfg.decay;
        Ok(json!({
            "path": self.path().display().to_string(),
            "file_bytes": file_bytes,
            "vector": {
                "dim": meta_get(&meta, keys::DIM)?,
                "source": meta_str_get(&txn.open_table(META_STR)?, keys::EMBEDDING_SOURCE)?,
                "embedder": self.embedder().source(),
                "embedder_spec": meta_str_get(&txn.open_table(META_STR)?, keys::EMBEDDER_SPEC)?,
            },
            "counts": {
                "chunks": chunks,
                "vectors": txn.open_table(VECTORS)?.len()?,
                "terms": txn.open_table(TERMS)?.len()?,
                "facts": txn.open_table(TRIPLES)?.len()?,
                "entity_links": txn.open_table(ENTITY_CHUNKS)?.len()?,
                "episodes": txn.open_table(EPISODES)?.len()?,
                "auth_tokens": txn.open_table(crate::tables::AUTH_TOKENS)?.len()?,
            },
            "index": index,
            "decay": {
                "enabled": d.enabled,
                "half_life_days": d.half_life_ms as f64 / crate::decay::DAY_MS as f64,
                "floor": d.floor,
                "boost": d.boost,
                "protect": [d.protect_lo, d.protect_hi],
                "only_tags": cfg.decay_tags,
            },
            "metrics": self.metrics().snapshot(),
        }))
    }

    /// Export every table as JSON lines (`{"type": ..., ...}`) to `w`.
    /// Returns the number of lines written.
    pub fn dump(&self, w: &mut dyn Write, include_vectors: bool) -> Result<u64> {
        let txn = self.db().begin_read()?;
        let mut lines = 0u64;
        let mut emit = |v: Value| -> Result<()> {
            serde_json::to_writer(&mut *w, &v).map_err(ser)?;
            w.write_all(b"\n")?;
            lines += 1;
            Ok(())
        };
        for item in txn.open_table(META)?.range(..)? {
            let (k, v) = item?;
            emit(json!({"type": "meta", "key": k.value(), "value": v.value()}))?;
        }
        for item in txn.open_table(META_STR)?.range(..)? {
            let (k, v) = item?;
            emit(json!({"type": "meta", "key": k.value(), "value": v.value()}))?;
        }
        let terms = txn.open_table(TERMS)?;
        let mut names: IdMap<String> = IdMap::default();
        for item in terms.range(..)? {
            let (_, v) = item?;
            let t: Term = archive::decode(v.value())?;
            names.insert(t.id, t.name.clone());
            emit(json!({"type": "term", "id": id_to_hex(t.id), "name": t.name,
                "kind": if t.kind == crate::model::TERM_PREDICATE { "predicate" } else { "entity" },
                "created_at": t.created_at, "updated_at": t.updated_at}))?;
        }
        let name = |id: u64| names.get(&id).cloned().unwrap_or_else(|| id_to_hex(id));
        let vectors = txn.open_table(VECTORS)?;
        for item in txn.open_table(CHUNKS)?.range(..)? {
            let (_, v) = item?;
            let c: Chunk = archive::decode(v.value())?;
            let mut rec = json!({"type": "chunk", "id": id_to_hex(c.id), "text": c.text, "tags": c.tags,
                "entities": c.entities.iter().map(|e| name(*e)).collect::<Vec<_>>(), "source": c.source,
                "created_at": c.created_at, "updated_at": c.updated_at, "hits": c.hits});
            if include_vectors && let Some(g) = vectors.get(c.id)? {
                rec["vector"] = json!(archive::le_to_f32s(g.value()));
            }
            emit(rec)?;
        }
        for item in txn.open_table(HNSW_NODES)?.range(..)? {
            let (k, v) = item?;
            let n: HnswNode = archive::decode(v.value())?;
            let layers: Vec<Vec<Value>> =
                n.layers.iter().map(|l| l.iter().map(|x| json!([id_to_hex(x.id), x.sim])).collect()).collect();
            emit(json!({"type": "hnsw_node", "id": id_to_hex(k.value()), "level": n.level, "layers": layers}))?;
        }
        for item in txn.open_table(TRIPLES)?.range(..)? {
            let (_, v) = item?;
            let t: Triple = archive::decode(v.value())?;
            emit(json!({"type": "fact", "subject": name(t.subject), "predicate": name(t.predicate),
                "object": name(t.object), "created_at": t.created_at, "updated_at": t.updated_at,
                "source_chunk": (t.source_chunk != 0).then(|| id_to_hex(t.source_chunk)), "hits": t.hits}))?;
        }
        for item in txn.open_table(EPISODES)?.range(..)? {
            let (_, v) = item?;
            let e: MemoryItem = archive::decode(v.value())?;
            emit(json!({"type": "event", "id": id_to_hex(e.id), "ts": e.ts, "kind": e.kind, "text": e.text,
                "chunk_id": (e.chunk_id != 0).then(|| id_to_hex(e.chunk_id))}))?;
        }
        Ok(lines)
    }
}
