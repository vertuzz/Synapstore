//! Layer 2 — knowledge graph of SPO triplets on redb B-trees (PRD §4).
//!
//! * `spo`: (subject, predicate) → [object ids]
//! * `pos`: (predicate, object) → [subject ids]   (reverse lookup)
//! * `ops`: (object, predicate) → [subject ids]   (in-edges for traversal)
//! * `triples`: (s, p, o) → provenance record
//!
//! Every write is an upsert: re-asserting a fact only refreshes metadata.

pub mod query;

pub use query::{BfsParams, BfsResult, Direction, Edge, GraphReader};

use crate::archive;
use crate::error::{Error, Result};
use crate::hash::{entity_id, normalize, predicate_id};
use crate::idlist;
use crate::model::{TERM_ENTITY, TERM_PREDICATE, Term, Triple};
use crate::tables::{ENTITY_CHUNKS, OPS, POS, Pair, SPO, TERMS, TRIPLES, Triplet};
use redb::{ReadableTable, Table, WriteTransaction};

pub const MAX_NAME_LEN: usize = 512;

/// All graph tables opened inside one write transaction.
pub struct GraphWriter<'t> {
    pub terms: Table<'t, u64, &'static [u8]>,
    pub spo: Table<'t, Pair, &'static [u8]>,
    pub pos: Table<'t, Pair, &'static [u8]>,
    pub ops: Table<'t, Pair, &'static [u8]>,
    pub triples: Table<'t, Triplet, &'static [u8]>,
    pub entity_chunks: Table<'t, u64, &'static [u8]>,
}

fn pair_add(t: &mut Table<'_, Pair, &'static [u8]>, key: Pair, id: u64) -> Result<bool> {
    let next = match t.get(key)? {
        Some(g) => idlist::with(g.value(), id),
        None => Some(idlist::from_ids([id])),
    };
    match next {
        Some(bytes) => {
            t.insert(key, bytes.as_slice())?;
            Ok(true)
        }
        None => Ok(false),
    }
}

fn pair_remove(t: &mut Table<'_, Pair, &'static [u8]>, key: Pair, id: u64) -> Result<()> {
    let next = t.get(key)?.and_then(|g| idlist::without(g.value(), id));
    if let Some(bytes) = next {
        if bytes.is_empty() {
            t.remove(key)?;
        } else {
            t.insert(key, bytes.as_slice())?;
        }
    }
    Ok(())
}

/// Validate a term name: non-empty after trimming, bounded length.
pub fn validate_name(what: &str, name: &str) -> Result<()> {
    let t = name.trim();
    if t.is_empty() {
        return Err(Error::InvalidInput(format!("{what} must not be empty")));
    }
    if t.chars().count() > MAX_NAME_LEN {
        return Err(Error::InvalidInput(format!("{what} longer than {MAX_NAME_LEN} chars")));
    }
    Ok(())
}

impl<'t> GraphWriter<'t> {
    pub fn open(txn: &'t WriteTransaction) -> Result<Self> {
        Ok(GraphWriter {
            terms: txn.open_table(TERMS)?,
            spo: txn.open_table(SPO)?,
            pos: txn.open_table(POS)?,
            ops: txn.open_table(OPS)?,
            triples: txn.open_table(TRIPLES)?,
            entity_chunks: txn.open_table(ENTITY_CHUNKS)?,
        })
    }

    /// Upsert a term; returns its id. Detects 64-bit hash collisions.
    pub fn upsert_term(&mut self, kind: u8, name: &str, now: u64) -> Result<u64> {
        validate_name(if kind == TERM_ENTITY { "entity" } else { "predicate" }, name)?;
        let id = if kind == TERM_PREDICATE { predicate_id(name) } else { entity_id(name) };
        let existing: Option<Term> = match self.terms.get(id)? {
            Some(g) => Some(archive::decode(g.value())?),
            None => None,
        };
        let term = match existing {
            Some(t) if normalize(&t.name) != normalize(name) || t.kind != kind => {
                return Err(Error::HashCollision { table: "terms", id });
            }
            Some(t) if t.updated_at >= now => return Ok(id),
            Some(t) => Term { updated_at: now, ..t },
            None => Term { id, name: name.trim().to_string(), kind, created_at: now, updated_at: now },
        };
        self.terms.insert(id, archive::encode(&term)?.as_slice())?;
        Ok(id)
    }

    /// Upsert a fact by names. Returns `(s, p, o, created)`.
    pub fn assert_fact(
        &mut self,
        s: &str,
        p: &str,
        o: &str,
        now: u64,
        source_chunk: u64,
    ) -> Result<(u64, u64, u64, bool)> {
        let s = self.upsert_term(TERM_ENTITY, s, now)?;
        let p = self.upsert_term(TERM_PREDICATE, p, now)?;
        let o = self.upsert_term(TERM_ENTITY, o, now)?;
        let created = self.upsert_triple(s, p, o, now, source_chunk)?;
        Ok((s, p, o, created))
    }

    /// Upsert a triple by ids; returns true if it did not exist.
    pub fn upsert_triple(&mut self, s: u64, p: u64, o: u64, now: u64, source: u64) -> Result<bool> {
        let existing: Option<Triple> = match self.triples.get((s, p, o))? {
            Some(g) => Some(archive::decode(g.value())?),
            None => None,
        };
        let created = existing.is_none();
        let rec = match existing {
            Some(t) => Triple {
                updated_at: t.updated_at.max(now),
                source_chunk: if source != 0 { source } else { t.source_chunk },
                hits: t.hits.saturating_add(1),
                ..t
            },
            None => Triple {
                subject: s,
                predicate: p,
                object: o,
                created_at: now,
                updated_at: now,
                source_chunk: source,
                hits: 1,
            },
        };
        self.triples.insert((s, p, o), archive::encode(&rec)?.as_slice())?;
        pair_add(&mut self.spo, (s, p), o)?;
        pair_add(&mut self.pos, (p, o), s)?;
        pair_add(&mut self.ops, (o, p), s)?;
        Ok(created)
    }

    /// Remove a fact from every index; returns true if it existed.
    pub fn retract(&mut self, s: u64, p: u64, o: u64) -> Result<bool> {
        if self.triples.remove((s, p, o))?.is_none() {
            return Ok(false);
        }
        pair_remove(&mut self.spo, (s, p), o)?;
        pair_remove(&mut self.pos, (p, o), s)?;
        pair_remove(&mut self.ops, (o, p), s)?;
        Ok(true)
    }

    /// Retract every fact whose provenance (`source_chunk`) is one of
    /// `chunks`; returns how many were removed. Facts asserted without a
    /// source are never touched. A full scan: forgetting is rare.
    pub fn retract_sourced(&mut self, chunks: &[u64]) -> Result<usize> {
        if chunks.is_empty() {
            return Ok(0);
        }
        let mut doomed = Vec::new();
        for row in self.triples.iter()? {
            let (k, v) = row?;
            let t: Triple = archive::decode(v.value())?;
            if t.source_chunk != 0 && chunks.contains(&t.source_chunk) {
                doomed.push(k.value());
            }
        }
        for &(s, p, o) in &doomed {
            self.retract(s, p, o)?;
        }
        Ok(doomed.len())
    }

    /// Record that `entity` is mentioned by `chunk`.
    pub fn link_chunk(&mut self, entity: u64, chunk: u64) -> Result<()> {
        let next = match self.entity_chunks.get(entity)? {
            Some(g) => idlist::with(g.value(), chunk),
            None => Some(idlist::from_ids([chunk])),
        };
        if let Some(bytes) = next {
            self.entity_chunks.insert(entity, bytes.as_slice())?;
        }
        Ok(())
    }

    pub fn unlink_chunk(&mut self, entity: u64, chunk: u64) -> Result<()> {
        let next = self.entity_chunks.get(entity)?.and_then(|g| idlist::without(g.value(), chunk));
        if let Some(bytes) = next {
            if bytes.is_empty() {
                self.entity_chunks.remove(entity)?;
            } else {
                self.entity_chunks.insert(entity, bytes.as_slice())?;
            }
        }
        Ok(())
    }
}
