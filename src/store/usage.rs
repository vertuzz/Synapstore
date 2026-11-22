//! Opt-in usage analytics (`serve --usage-log`): which client called which
//! tool, what was searched, how well it matched, and which memories are
//! recalled (or never). `synapstore usage` summarizes it.

use super::SynapStore;
use super::apply::WriteOp;
use crate::archive;
use crate::error::{Error, Result};
use crate::hash::{hex_to_id, id_to_hex, xxh64};
use crate::model::Chunk;
use crate::tables::{CHUNKS, RECALL_STATS, USAGE_LOG};
use crate::timefmt::rfc3339;
use redb::{ReadableDatabase, ReadableTable, ReadableTableMetadata, WriteTransaction};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Rows kept in `usage_log`; older ones are pruned.
pub const MAX_ROWS: u64 = 20_000;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageRecord {
    pub ts: u64,
    pub client: String,
    pub tool: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    /// Hex ids of the memories returned (recall only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hits: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_similarity: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<String>,
    #[serde(default)]
    pub ok: bool,
}

pub fn apply(txn: &WriteTransaction, r: &UsageRecord) -> Result<()> {
    let bytes = serde_json::to_vec(r).map_err(|e| Error::Serialization(e.to_string()))?;
    let mut log = txn.open_table(USAGE_LOG)?;
    log.insert((r.ts, xxh64(&bytes, 0x5553)), bytes.as_slice())?;
    while log.len()? > MAX_ROWS {
        if log.pop_first()?.is_none() {
            break;
        }
    }
    let mut stats = txn.open_table(RECALL_STATS)?;
    for h in &r.hits {
        let Some(id) = hex_to_id(h) else { continue };
        let n = stats.get(id)?.map_or(0, |g| g.value().0);
        stats.insert(id, (n + 1, r.ts))?;
    }
    Ok(())
}

fn preview(c: &Chunk) -> String {
    c.text.chars().take(100).collect()
}

impl SynapStore {
    /// Record a tool call in the background (never slows the caller).
    pub fn record_usage(&self, r: UsageRecord) {
        let s = self.clone();
        let spawned = std::thread::Builder::new().name("synapstore-usage".into()).spawn(move || {
            if let Err(e) = s.write(WriteOp::Usage(Box::new(r))) {
                s.log().warn("usage_log", &[("error", e.to_string())]);
            }
        });
        if let Err(e) = spawned {
            self.log().warn("usage_log", &[("error", e.to_string())]);
        }
    }

    /// Summary of the usage log (see [`UsageRecord`]).
    pub fn usage(&self, limit: usize) -> Result<Value> {
        let txn = self.db().begin_read()?;
        let mut rows = Vec::new();
        for item in txn.open_table(USAGE_LOG)?.iter()? {
            if let Ok(r) = serde_json::from_slice::<UsageRecord>(item?.1.value()) {
                rows.push(r);
            }
        }
        let mut by_client: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
        for r in &rows {
            *by_client.entry(r.client.clone()).or_default().entry(r.tool.clone()).or_default() += 1;
        }
        let mut recalls: Vec<&UsageRecord> = rows.iter().filter(|r| r.tool == "recall").collect();
        let fallbacks = recalls.iter().filter(|r| r.embedding.as_deref() == Some(super::fallback::LEXICAL_FALLBACK));
        let fallbacks = fallbacks.count();
        let empty = recalls.iter().filter(|r| r.ok && r.hits.is_empty()).count();
        recalls.sort_by(|a, b| a.top_similarity.unwrap_or(0.0).total_cmp(&b.top_similarity.unwrap_or(0.0)));
        let weakest: Vec<Value> = recalls
            .iter()
            .take(limit)
            .map(|r| json!({"time": rfc3339(r.ts), "client": r.client, "query": r.query, "top_similarity": r.top_similarity}))
            .collect();
        let (chunks, stats) = (txn.open_table(CHUNKS)?, txn.open_table(RECALL_STATS)?);
        let mut counted = Vec::new();
        for item in stats.iter()? {
            let (k, v) = item?;
            counted.push((k.value(), v.value()));
        }
        counted.sort_by(|a, b| b.1.0.cmp(&a.1.0));
        let mut most = Vec::new();
        for (id, (n, last)) in counted.iter() {
            if most.len() >= limit {
                break;
            }
            if let Some(g) = chunks.get(*id)? {
                let c = archive::decode::<Chunk>(g.value())?;
                most.push(json!({"id": id_to_hex(*id), "recalled": n, "last": rfc3339(*last), "text": preview(&c)}));
            }
        }
        let mut never = Vec::new();
        for item in chunks.iter()? {
            let (k, v) = item?;
            if stats.get(k.value())?.is_none() {
                never.push(archive::decode::<Chunk>(v.value())?);
            }
        }
        never.sort_by_key(|c| c.updated_at);
        let examples: Vec<Value> = never
            .iter()
            .take(limit)
            .map(|c| json!({"id": id_to_hex(c.id), "updated": rfc3339(c.updated_at), "text": preview(c)}))
            .collect();
        Ok(json!({
            "calls": rows.len(),
            "since": rows.first().map(|r| rfc3339(r.ts)),
            "by_client": by_client,
            "recall": {"count": recalls.len(), "lexical_fallbacks": fallbacks, "empty": empty, "weakest": weakest},
            "most_recalled": most,
            "never_recalled": {"count": never.len(), "of": chunks.len()?, "oldest": examples},
        }))
    }
}
