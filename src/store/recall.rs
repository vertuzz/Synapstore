//! `recall`: HNSW candidate generation + time-decay re-ranking, reading
//! chunk records zero-copy from the snapshot.

use super::SynapStore;
use super::types::{RecallHit, RecallOutcome, RecallQuery};
use crate::archive;
use crate::decay::{self, DAY_MS, DecayParams};
use crate::embed::{self, Embedder, SOURCE_EXTERNAL};
use crate::error::{Error, Result};
use crate::hnsw::{self, EntryPoint, Scored, View};
use crate::math;
use crate::metrics::{Metrics, Op};
use crate::model::{Chunk, Term};
use crate::tables::{CHUNKS, HNSW_NODES, META, META_STR, TERMS, VECTORS, keys, meta_get, meta_str_get};
use crate::timefmt::now_ms;
use redb::{ReadableDatabase, ReadableTable};

/// Upper bound for the adaptive beam width.
pub const MAX_EF: usize = 4096;

/// Resolve the query into a unit vector living in the store's space.
pub fn query_vector(
    vector: Option<&[f32]>,
    text: Option<&str>,
    dim: usize,
    store_src: &str,
    embedder: &dyn Embedder,
) -> Result<Vec<f32>> {
    let unit = match (vector, text) {
        (Some(v), _) => {
            if store_src != SOURCE_EXTERNAL {
                return Err(Error::EmbeddingMismatch { store: store_src.into(), request: SOURCE_EXTERNAL.into() });
            }
            math::normalize(v)?
        }
        (None, Some(t)) if !t.trim().is_empty() => {
            if store_src != embedder.source() {
                return Err(Error::EmbeddingMismatch { store: store_src.into(), request: embedder.source().into() });
            }
            embed::query_in_space(embedder, t, dim)?
        }
        _ => {
            return Err(Error::InvalidInput("provide `vector` or a non-empty `query` text".into()));
        }
    };
    if unit.len() != dim {
        return Err(Error::DimensionMismatch { expected: dim, got: unit.len() });
    }
    Ok(unit)
}

/// A candidate after time-decay scoring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ranked {
    pub id: u64,
    pub sim: f32,
    pub score: f32,
    pub factor: f32,
    pub age_ms: u64,
}

/// How candidates are filtered and scored.
#[derive(Debug, Clone, Copy)]
pub struct RankOpts<'a> {
    pub now: u64,
    pub decay: &'a DecayParams,
    /// Only memories with one of these tags decay (empty = all).
    pub decay_tags: &'a [String],
    pub max_age_ms: Option<u64>,
    /// Keep only memories with one of these tags (empty = all).
    pub tags: &'a [String],
}

/// Filter (tombstones, `max_age`, tags) and score candidates; sorted by score.
pub fn rank<C>(chunks: &C, cands: &[Scored], o: &RankOpts<'_>, metrics: &Metrics) -> Result<Vec<Ranked>>
where
    C: ReadableTable<u64, &'static [u8]>,
{
    let mut out = Vec::with_capacity(cands.len());
    for c in cands {
        let Some(g) = chunks.get(c.id)? else { continue };
        let chunk = archive::view::<Chunk>(g.value())?;
        Metrics::add(&metrics.counters.zero_copy_reads, 1);
        let age_ms = decay::age(o.now, chunk.updated_at.to_native());
        if o.max_age_ms.is_some_and(|m| age_ms > m) {
            continue;
        }
        if !o.tags.is_empty() && !chunk.tags.iter().any(|t| o.tags.iter().any(|w| w.eq_ignore_ascii_case(t.as_str()))) {
            continue;
        }
        let p = decay::params_for(o.decay, decay::decays(o.decay_tags, chunk.tags.iter().map(|t| t.as_str())));
        let factor = decay::factor(c.sim, age_ms, &p);
        out.push(Ranked { id: c.id, sim: c.sim, score: decay::score(c.sim, age_ms, &p), factor, age_ms });
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
    Ok(out)
}

pub(super) fn hit<C, T>(chunks: &C, terms: &T, r: &Ranked) -> Result<Option<RecallHit>>
where
    C: ReadableTable<u64, &'static [u8]>,
    T: ReadableTable<u64, &'static [u8]>,
{
    let Some(g) = chunks.get(r.id)? else {
        return Ok(None);
    };
    let c = archive::view::<Chunk>(g.value())?;
    let mut entities = Vec::with_capacity(c.entities.len());
    for e in c.entities.iter() {
        if let Some(tg) = terms.get(e.to_native())? {
            entities.push(archive::view::<Term>(tg.value())?.name.as_str().to_string());
        }
    }
    Ok(Some(RecallHit {
        id: r.id,
        text: c.text.as_str().to_string(),
        tags: c.tags.iter().map(|t| t.as_str().to_string()).collect(),
        entities,
        source: c.source.as_str().to_string(),
        similarity: r.sim,
        score: r.score,
        decay_factor: r.factor,
        age_days: r.age_ms as f64 / DAY_MS as f64,
        created_at: c.created_at.to_native(),
        updated_at: c.updated_at.to_native(),
        hits: c.hits.to_native(),
    }))
}

impl SynapStore {
    /// Semantic recall with time-decay (PRD `recall(vector, top_k, max_age)`).
    pub fn recall(&self, q: &RecallQuery) -> Result<RecallOutcome> {
        self.timed(Op::Recall, || self.recall_inner(q))
    }

    fn recall_inner(&self, q: &RecallQuery) -> Result<RecallOutcome> {
        // Validate the query even when the store is still empty.
        match (&q.vector, &q.text) {
            (Some(v), _) => drop(math::normalize(v)?),
            (None, Some(t)) if !t.trim().is_empty() => {}
            _ => {
                return Err(Error::InvalidInput("provide `vector` or a non-empty `query` text".into()));
            }
        }
        let txn = self.db().begin_read()?;
        let meta = txn.open_table(META)?;
        let (Some(dim), Some(src)) =
            (meta_get(&meta, keys::DIM)?, meta_str_get(&txn.open_table(META_STR)?, keys::EMBEDDING_SOURCE)?)
        else {
            return Ok(RecallOutcome { embedding: "none".into(), ..Default::default() });
        };
        let unit = match query_vector(q.vector.as_deref(), q.text.as_deref(), dim as usize, &src, self.embedder()) {
            Ok(u) => u,
            // The embedding endpoint is down: answer from words rather than fail.
            Err(Error::Embedding(why)) => return self.lexical_fallback(q, &txn, why),
            Err(e) => return Err(e),
        };
        let Some(ep) = EntryPoint::load(&meta)? else {
            return Ok(RecallOutcome { embedding: src, ..Default::default() });
        };
        let (vectors, nodes) = (txn.open_table(VECTORS)?, txn.open_table(HNSW_NODES)?);
        let (chunks, terms) = (txn.open_table(CHUNKS)?, txn.open_table(TERMS)?);
        let view = View::new(&vectors, &nodes);
        let cfg = self.config();
        let top_k = q.top_k.clamp(1, 1000);
        let decay_p = q.decay.unwrap_or(cfg.decay);
        let now = q.now_ms.unwrap_or_else(now_ms);
        let mut ef = q.ef.unwrap_or(cfg.hnsw.ef_search).max(top_k * 4).min(MAX_EF);
        let mut visited = 0;
        let counters = &self.metrics().counters;
        let (ranked, ncand) = loop {
            let cands = hnsw::knn(&view, ep, &unit, top_k, ef, &mut visited)?;
            Metrics::add(&counters.hnsw_searches, 1);
            let opts =
                RankOpts { now, decay: &decay_p, decay_tags: &cfg.decay_tags, max_age_ms: q.max_age_ms, tags: &q.tags };
            let ranked = rank(&chunks, &cands, &opts, self.metrics())?;
            // Filters may starve the result: widen the beam and retry.
            if ranked.len() >= top_k || cands.len() < ef || ef >= MAX_EF {
                break (ranked, cands.len());
            }
            ef = (ef * 4).min(MAX_EF);
        };
        Metrics::add(&counters.hnsw_nodes_visited, visited);
        let mut hits = Vec::with_capacity(top_k);
        for r in ranked.iter().take(top_k) {
            if let Some(h) = hit(&chunks, &terms, r)? {
                hits.push(h);
            }
        }
        Ok(RecallOutcome { hits, candidates: ncand, visited, ef, embedding: src, warning: None })
    }
}
