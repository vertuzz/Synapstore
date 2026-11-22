//! Verbatim import of records into a store (used by `reindex`): memories
//! keep their timestamps and counters, facts their provenance, events and
//! API tokens are copied as they are. No episode is appended for imports.

use super::SynapStore;
use super::apply::{ApplyCtx, WriteOp, WriteResult, append_event, check_space};
use super::prepare::{PreparedChunk, PreparedEvent};
use crate::archive::{decode, encode};
use crate::error::{Error, Result};
use crate::graph::GraphWriter;
use crate::hnsw;
use crate::metrics::Metrics;
use crate::model::{Chunk, TERM_ENTITY, Triple};
use crate::tables::{AUTH_TOKENS, CHUNKS, HNSW_NODES, META, VECTORS};
use redb::{ReadableTable, WriteTransaction};

/// One memory with its original metadata, already chunked and embedded.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportMemory {
    pub chunks: Vec<PreparedChunk>,
    pub embedding: String,
    pub tags: Vec<String>,
    pub entities: Vec<String>,
    pub source: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub hits: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub source_chunk: u64,
    pub hits: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImportOp {
    Memory(Box<ImportMemory>),
    Fact(ImportFact),
    Event(PreparedEvent),
    /// An `auth_tokens` row (key = SHA-256 of the secret), copied raw.
    Token {
        key: Vec<u8>,
        value: Vec<u8>,
    },
}

fn memory(txn: &WriteTransaction, m: &ImportMemory, ctx: &ApplyCtx<'_>) -> Result<()> {
    let Some(first) = m.chunks.first() else { return Ok(()) };
    check_space(txn, first.unit.len(), &m.embedding)?;
    let mut g = GraphWriter::open(txn)?;
    let ids = m.entities.iter().map(|e| g.upsert_term(TERM_ENTITY, e, m.created_at)).collect::<Result<Vec<u64>>>()?;
    {
        let mut chunks = txn.open_table(CHUNKS)?;
        let mut vectors = txn.open_table(VECTORS)?;
        let mut nodes = txn.open_table(HNSW_NODES)?;
        let mut meta = txn.open_table(META)?;
        for c in &m.chunks {
            let old: Option<Chunk> = chunks.get(c.id)?.map(|g| decode(g.value())).transpose()?;
            let rec = match old {
                Some(o) if o.text != c.text => return Err(Error::HashCollision { table: "chunks", id: c.id }),
                Some(o) => Chunk {
                    tags: union(o.tags, &m.tags),
                    created_at: o.created_at.min(m.created_at),
                    updated_at: o.updated_at.max(m.updated_at),
                    hits: o.hits.max(m.hits),
                    ..o
                },
                None => Chunk {
                    id: c.id,
                    text: c.text.clone(),
                    tags: m.tags.clone(),
                    entities: ids.clone(),
                    source: m.source.clone(),
                    created_at: m.created_at,
                    updated_at: m.updated_at,
                    hits: m.hits,
                },
            };
            chunks.insert(c.id, encode(&rec)?.as_slice())?;
            hnsw::insert::insert_cached(&mut vectors, &mut nodes, &mut meta, ctx.hnsw, c.id, &c.unit, ctx.cache)?;
            Metrics::add(&ctx.metrics.counters.hnsw_inserts, 1);
        }
        if ctx.cache.dirty_len() >= hnsw::cache::FLUSH_AT {
            ctx.cache.flush(&mut nodes)?;
        }
    }
    for c in &m.chunks {
        for e in &ids {
            g.link_chunk(*e, c.id)?;
        }
    }
    Ok(())
}

fn union(mut a: Vec<String>, b: &[String]) -> Vec<String> {
    for t in b {
        if !a.iter().any(|x| x.eq_ignore_ascii_case(t)) {
            a.push(t.clone());
        }
    }
    a
}

fn fact(txn: &WriteTransaction, f: &ImportFact) -> Result<()> {
    let mut g = GraphWriter::open(txn)?;
    let (s, p, o, _) = g.assert_fact(&f.subject, &f.predicate, &f.object, f.created_at, f.source_chunk)?;
    let rec = Triple {
        subject: s,
        predicate: p,
        object: o,
        created_at: f.created_at,
        updated_at: f.updated_at.max(f.created_at),
        source_chunk: f.source_chunk,
        hits: f.hits.max(1),
    };
    g.triples.insert((s, p, o), encode(&rec)?.as_slice())?;
    Ok(())
}

/// Apply imports inside the writer's transaction; returns how many ran.
pub fn apply(txn: &WriteTransaction, ops: &[ImportOp], ctx: &ApplyCtx<'_>) -> Result<usize> {
    for op in ops {
        match op {
            ImportOp::Memory(m) => memory(txn, m, ctx)?,
            ImportOp::Fact(f) => fact(txn, f)?,
            ImportOp::Event(e) => append_event(txn, e)?,
            ImportOp::Token { key, value } => {
                txn.open_table(AUTH_TOKENS)?.insert(key.as_slice(), value.as_slice())?;
            }
        }
    }
    Ok(ops.len())
}

impl SynapStore {
    /// Import records verbatim in one transaction (see [`ImportOp`]).
    pub fn import(&self, ops: Vec<ImportOp>) -> Result<usize> {
        match self.write(WriteOp::Import(ops))? {
            WriteResult::Import(n) => Ok(n),
            other => Err(Error::Storage(format!("unexpected writer result {other:?}"))),
        }
    }
}
