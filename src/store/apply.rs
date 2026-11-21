//! Write operations, applied inside a redb write transaction by the single
//! writer thread. All writes are upserts (PRD §6.3).

use super::prepare::{PreparedEvent, PreparedMemorize};
use super::types::{FactInput, ForgetOutcome, MemorizeOutcome};
use crate::archive::{decode, encode};
use crate::error::{Error, Result};
use crate::graph::GraphWriter;
use crate::hash::{entity_id, normalize, predicate_id};
use crate::hnsw::cache::WriteCache;
use crate::hnsw::repair::{self, RepairOutcome};
use crate::hnsw::{self, HnswParams};
use crate::metrics::Metrics;
use crate::model::{Chunk, MemoryItem, TERM_ENTITY};
use crate::tables::{CHUNKS, EPISODES, HNSW_NODES, META, META_STR, VECTORS, keys, meta_get, meta_str_get};
use redb::{ReadableTable, WriteTransaction};

/// A unit of work for the writer thread.
#[derive(Debug, Clone)]
pub enum WriteOp {
    Memorize(Box<PreparedMemorize>),
    Facts {
        facts: Vec<FactInput>,
        ts: u64,
        source: u64,
    },
    Retract(FactInput),
    Events(Vec<PreparedEvent>),
    Forget(Vec<u64>),
    /// Reconnect unreachable HNSW nodes.
    Repair,
    /// API-token administration.
    Auth(crate::auth::AuthOp),
    /// Several operations committed atomically together.
    Batch(Vec<WriteOp>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum WriteResult {
    Memorize(MemorizeOutcome),
    Facts(usize),
    Retract(bool),
    Events(Vec<u64>),
    Forget(ForgetOutcome),
    Repair(RepairOutcome),
    Auth(crate::auth::AuthResult),
    Batch(Vec<WriteResult>),
}

pub struct ApplyCtx<'a> {
    pub hnsw: &'a HnswParams,
    pub metrics: &'a Metrics,
    /// Transaction-wide HNSW write-back cache; flushed before commit.
    pub cache: &'a WriteCache,
}

/// Write cached HNSW nodes back and drop the cache (before code that
/// reads or writes the HNSW tables directly).
fn sync_cache(txn: &WriteTransaction, ctx: &ApplyCtx<'_>) -> Result<()> {
    ctx.cache.flush(&mut txn.open_table(HNSW_NODES)?)?;
    ctx.cache.clear();
    Ok(())
}

pub fn apply(txn: &WriteTransaction, op: &WriteOp, ctx: &ApplyCtx<'_>) -> Result<WriteResult> {
    match op {
        WriteOp::Memorize(m) => apply_memorize(txn, m, ctx).map(WriteResult::Memorize),
        WriteOp::Facts { facts, ts, source } => apply_facts(txn, facts, *ts, *source).map(WriteResult::Facts),
        WriteOp::Retract(f) => {
            let mut g = GraphWriter::open(txn)?;
            let r = g.retract(entity_id(&f.subject), predicate_id(&f.predicate), entity_id(&f.object))?;
            Ok(WriteResult::Retract(r))
        }
        WriteOp::Events(evs) => {
            for e in evs {
                append_event(txn, e)?;
            }
            Metrics::add(&ctx.metrics.counters.events_logged, evs.len() as u64);
            Ok(WriteResult::Events(evs.iter().map(|e| e.id).collect()))
        }
        WriteOp::Forget(ids) => {
            sync_cache(txn, ctx)?;
            apply_forget(txn, ids).map(WriteResult::Forget)
        }
        WriteOp::Auth(op) => crate::auth::ops::apply(txn, op).map(WriteResult::Auth),
        WriteOp::Repair => {
            sync_cache(txn, ctx)?;
            let vectors = txn.open_table(VECTORS)?;
            let mut nodes = txn.open_table(HNSW_NODES)?;
            let ep = hnsw::EntryPoint::load(&txn.open_table(META)?)?;
            let o = repair::repair(&vectors, &mut nodes, ep, ctx.hnsw)?;
            note_repair(ctx.metrics, &o);
            Ok(WriteResult::Repair(o))
        }
        WriteOp::Batch(ops) => {
            ops.iter().map(|o| apply(txn, o, ctx)).collect::<Result<Vec<_>>>().map(WriteResult::Batch)
        }
    }
}

/// Pin the store's vector space (embedding source + dimension) on first use.
/// The source is checked first: it is the more actionable error.
fn check_space(txn: &WriteTransaction, dim: usize, source: &str) -> Result<()> {
    let mut ms = txn.open_table(META_STR)?;
    match meta_str_get(&ms, keys::EMBEDDING_SOURCE)? {
        Some(s) if s != source => {
            return Err(Error::EmbeddingMismatch { store: s, request: source.into() });
        }
        Some(_) => {}
        None => {
            ms.insert(keys::EMBEDDING_SOURCE, source)?;
        }
    }
    let mut meta = txn.open_table(META)?;
    match meta_get(&meta, keys::DIM)? {
        Some(d) if d as usize != dim => Err(Error::DimensionMismatch { expected: d as usize, got: dim }),
        Some(_) => Ok(()),
        None => {
            meta.insert(keys::DIM, dim as u64)?;
            Ok(())
        }
    }
}

fn union_into(dst: &mut Vec<String>, src: &[String]) {
    for s in src {
        if !dst.iter().any(|d| d.eq_ignore_ascii_case(s)) {
            dst.push(s.clone());
        }
    }
}

fn merge_chunk(old: Chunk, m: &PreparedMemorize, entity_ids: &[u64]) -> Chunk {
    let mut tags = old.tags;
    union_into(&mut tags, &m.tags);
    let mut entities = old.entities;
    for e in entity_ids {
        if !entities.contains(e) {
            entities.push(*e);
        }
    }
    Chunk {
        tags,
        entities,
        source: if m.source.is_empty() { old.source } else { m.source.clone() },
        created_at: old.created_at.min(m.ts),
        updated_at: old.updated_at.max(m.ts),
        hits: old.hits.saturating_add(1),
        ..old
    }
}

fn apply_memorize(txn: &WriteTransaction, m: &PreparedMemorize, ctx: &ApplyCtx<'_>) -> Result<MemorizeOutcome> {
    let dim = m.chunks.first().map_or(0, |c| c.unit.len());
    let mut out =
        MemorizeOutcome { entities: m.entities.clone(), embedding: m.embedding.clone(), dim, ..Default::default() };
    if m.chunks.is_empty() {
        return Ok(out);
    }
    check_space(txn, dim, &m.embedding)?;
    let mut g = GraphWriter::open(txn)?;
    let entity_ids = m.entities.iter().map(|e| g.upsert_term(TERM_ENTITY, e, m.ts)).collect::<Result<Vec<u64>>>()?;
    let counters = &ctx.metrics.counters;
    {
        let mut chunks = txn.open_table(CHUNKS)?;
        let mut vectors = txn.open_table(VECTORS)?;
        let mut nodes = txn.open_table(HNSW_NODES)?;
        let mut meta = txn.open_table(META)?;
        for c in &m.chunks {
            let existing: Option<Chunk> = chunks.get(c.id)?.map(|g| decode(g.value())).transpose()?;
            let rec = match existing {
                Some(old) if normalize(&old.text) != normalize(&c.text) => {
                    Metrics::add(&counters.hash_collisions, 1);
                    return Err(Error::HashCollision { table: "chunks", id: c.id });
                }
                Some(old) => {
                    out.updated += 1;
                    merge_chunk(old, m, &entity_ids)
                }
                None => {
                    out.created += 1;
                    Chunk {
                        id: c.id,
                        text: c.text.clone(),
                        tags: m.tags.clone(),
                        entities: entity_ids.clone(),
                        source: m.source.clone(),
                        created_at: m.ts,
                        updated_at: m.ts,
                        hits: 1,
                    }
                }
            };
            chunks.insert(c.id, encode(&rec)?.as_slice())?;
            let ins =
                hnsw::insert::insert_cached(&mut vectors, &mut nodes, &mut meta, ctx.hnsw, c.id, &c.unit, ctx.cache)?;
            Metrics::add(&counters.hnsw_nodes_visited, ins.visited);
            if ins.unchanged {
                out.unchanged += 1;
            } else {
                Metrics::add(&counters.hnsw_inserts, 1);
            }
            if ins.relinked {
                Metrics::add(&counters.hnsw_relinks, 1);
            }
            out.chunk_ids.push(c.id);
        }
        let inserted = (out.chunk_ids.len() - out.unchanged) as u64;
        if ctx.cache.dirty_len() >= crate::hnsw::cache::FLUSH_AT {
            ctx.cache.flush(&mut nodes)?;
        }
        if let Some(o) = repair::maybe_repair(&vectors, &mut nodes, &mut meta, ctx.hnsw, inserted, Some(ctx.cache))? {
            note_repair(ctx.metrics, &o);
        }
    }
    Metrics::add(&counters.chunks_created, out.created as u64);
    Metrics::add(&counters.chunks_updated, out.updated as u64);
    for c in &m.chunks {
        for e in &entity_ids {
            g.link_chunk(*e, c.id)?;
        }
    }
    let source = m.chunks[0].id;
    for f in &m.facts {
        let (.., created) = g.assert_fact(&f.subject, &f.predicate, &f.object, m.ts, source)?;
        out.facts_added += usize::from(created);
    }
    Metrics::add(&counters.triples_created, out.facts_added as u64);
    drop(g);
    append_event(txn, &m.event)?;
    Ok(out)
}

fn note_repair(m: &Metrics, o: &RepairOutcome) {
    Metrics::add(&m.counters.hnsw_repairs, 1);
    Metrics::add(&m.counters.hnsw_repair_links, o.links_added);
}

fn apply_facts(txn: &WriteTransaction, facts: &[FactInput], ts: u64, source: u64) -> Result<usize> {
    let mut g = GraphWriter::open(txn)?;
    let mut created = 0;
    for f in facts {
        let (s, _, o, c) = g.assert_fact(&f.subject, &f.predicate, &f.object, ts, source)?;
        if source != 0 {
            g.link_chunk(s, source)?;
            g.link_chunk(o, source)?;
        }
        created += usize::from(c);
    }
    Ok(created)
}

fn append_event(txn: &WriteTransaction, e: &PreparedEvent) -> Result<()> {
    let mut t = txn.open_table(EPISODES)?;
    let item = MemoryItem { ts: e.ts, id: e.id, kind: e.kind.clone(), text: e.text.clone(), chunk_id: e.chunk_id };
    t.insert((e.ts, e.id), encode(&item)?.as_slice())?;
    Ok(())
}

/// Delete chunks and the facts they asserted. Their HNSW nodes stay as
/// routing tombstones (the vector is kept so the graph remains navigable);
/// recall skips them.
fn apply_forget(txn: &WriteTransaction, ids: &[u64]) -> Result<ForgetOutcome> {
    let mut removed = Vec::new();
    {
        let mut chunks = txn.open_table(CHUNKS)?;
        for id in ids {
            let old = chunks.remove(*id)?.map(|g| decode::<Chunk>(g.value())).transpose()?;
            if let Some(c) = old {
                removed.push(c);
            }
        }
    }
    let mut g = GraphWriter::open(txn)?;
    for c in &removed {
        for e in &c.entities {
            g.unlink_chunk(*e, c.id)?;
        }
    }
    let facts_retracted = g.retract_sourced(&removed.iter().map(|c| c.id).collect::<Vec<_>>())?;
    Ok(ForgetOutcome { forgotten: removed.len(), facts_retracted })
}
