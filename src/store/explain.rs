//! Whitebox tracing (PRD §9): explain *why* a given memory was or was not
//! returned for a query — exact rank, HNSW reachability, decay, filters.

use super::SynapStore;
use super::recall::{RankOpts, query_vector, rank};
use super::types::hex;
use crate::archive;
use crate::decay::{self, DAY_MS};
use crate::error::Result;
use crate::hnsw::{self, EntryPoint, View};
use crate::math;
use crate::metrics::Op;
use crate::model::Chunk;
use crate::tables::{CHUNKS, HNSW_NODES, META, META_STR, VECTORS, keys, meta_get, meta_str_get};
use crate::timefmt::now_ms;
use redb::{ReadableDatabase, ReadableTable, ReadableTableMetadata};
use serde::Serialize;

#[derive(Debug, Clone, Default)]
pub struct ExplainQuery {
    pub target: u64,
    pub vector: Option<Vec<f32>>,
    pub text: Option<String>,
    pub top_k: usize,
    pub ef: Option<usize>,
    pub max_age_ms: Option<u64>,
    pub now_ms: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Explanation {
    #[serde(serialize_with = "hex::one")]
    pub target: u64,
    pub stored: bool,
    pub indexed: bool,
    pub text: Option<String>,
    pub node_level: Option<u8>,
    pub node_degree: Vec<usize>,
    pub exact_similarity: Option<f32>,
    /// 1-based rank among all live memories by raw similarity.
    pub exact_rank: Option<usize>,
    /// 1-based position inside the HNSW candidate list (None = not reached).
    pub hnsw_rank: Option<usize>,
    pub hnsw_candidates: usize,
    pub ef: usize,
    pub age_days: Option<f64>,
    pub decay_factor: Option<f32>,
    pub score: Option<f32>,
    /// 1-based rank after time-decay re-ranking.
    pub final_rank: Option<usize>,
    pub top_k: usize,
    pub filtered_by_max_age: bool,
    pub verdict: String,
}

impl SynapStore {
    /// Exact top-`k` live memories by raw cosine similarity (brute force).
    /// Ground truth for benchmarks and whitebox debugging.
    pub fn exact_search(&self, q: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        let unit = math::normalize(q)?;
        let txn = self.db().begin_read()?;
        let (chunks, vectors) = (txn.open_table(CHUNKS)?, txn.open_table(VECTORS)?);
        let tombstones = vectors.len()?.saturating_sub(chunks.len()?) as usize;
        let all = crate::hnsw::inspect::brute_force(&vectors, &unit, k + tombstones)?;
        let mut out = Vec::with_capacity(k);
        for s in all {
            if out.len() < k && chunks.get(s.id)?.is_some() {
                out.push((s.id, s.sim));
            }
        }
        Ok(out)
    }

    pub fn explain(&self, q: &ExplainQuery) -> Result<Explanation> {
        self.timed(Op::Explain, || self.explain_inner(q))
    }

    fn explain_inner(&self, q: &ExplainQuery) -> Result<Explanation> {
        let top_k = q.top_k.clamp(1, 1000);
        let mut ex = Explanation { target: q.target, top_k, ..Default::default() };
        let txn = self.db().begin_read()?;
        let (chunks, vectors, nodes) = (txn.open_table(CHUNKS)?, txn.open_table(VECTORS)?, txn.open_table(HNSW_NODES)?);
        let meta = txn.open_table(META)?;
        if let Some(g) = chunks.get(q.target)? {
            ex.stored = true;
            ex.text = Some(archive::view::<Chunk>(g.value())?.text.as_str().to_string());
        }
        let view = View::new(&vectors, &nodes);
        if let Some(node) = view.node(q.target)? {
            ex.indexed = true;
            ex.node_level = Some(node.level);
            ex.node_degree = node.layers.iter().map(Vec::len).collect();
        }
        let (Some(dim), Some(src)) =
            (meta_get(&meta, keys::DIM)?, meta_str_get(&txn.open_table(META_STR)?, keys::EMBEDDING_SOURCE)?)
        else {
            ex.verdict = "store_is_empty".into();
            return Ok(ex);
        };
        let unit = query_vector(q.vector.as_deref(), q.text.as_deref(), dim as usize, &src, self.embedder())?;
        if let Some(g) = vectors.get(q.target)? {
            let sim = math::dot_le(&unit, g.value());
            ex.exact_similarity = Some(sim);
            let mut better = 0;
            for item in vectors.range(..)? {
                let (k, v) = item?;
                if k.value() != q.target && math::dot_le(&unit, v.value()) > sim && chunks.get(k.value())?.is_some() {
                    better += 1;
                }
            }
            ex.exact_rank = Some(better + 1);
        }
        let now = q.now_ms.unwrap_or_else(now_ms);
        let decay_p = self.config().decay;
        if let Some(g) = chunks.get(q.target)? {
            let c = archive::view::<Chunk>(g.value())?;
            let age = decay::age(now, c.updated_at.to_native());
            let decay_p = decay::params_for(
                &decay_p,
                decay::decays(&self.config().decay_tags, c.tags.iter().map(|t| t.as_str())),
            );
            let sim = ex.exact_similarity.unwrap_or(0.0);
            ex.age_days = Some(age as f64 / DAY_MS as f64);
            ex.decay_factor = Some(decay::factor(sim, age, &decay_p));
            ex.score = Some(decay::score(sim, age, &decay_p));
            ex.filtered_by_max_age = q.max_age_ms.is_some_and(|m| age > m);
        }
        ex.ef = q.ef.unwrap_or(self.config().hnsw.ef_search).max(top_k * 4);
        if let Some(ep) = EntryPoint::load(&meta)? {
            let mut visited = 0;
            let cands = hnsw::knn(&view, ep, &unit, top_k, ex.ef, &mut visited)?;
            ex.hnsw_candidates = cands.len();
            ex.hnsw_rank = cands.iter().position(|c| c.id == q.target).map(|p| p + 1);
            let decay_tags = &self.config().decay_tags;
            let opts = RankOpts { now, decay: &self.config().decay, decay_tags, max_age_ms: q.max_age_ms, tags: &[] };
            let ranked = rank(&chunks, &cands, &opts, self.metrics())?;
            ex.final_rank = ranked.iter().position(|r| r.id == q.target).map(|p| p + 1);
        }
        ex.verdict = verdict(&ex);
        Ok(ex)
    }
}

/// Human/machine-readable diagnosis (pure).
pub fn verdict(ex: &Explanation) -> String {
    let k = ex.top_k;
    if !ex.stored {
        return "not_stored: no live memory with this id (never memorized or forgotten)".into();
    }
    if !ex.indexed {
        return "not_indexed: memory has no HNSW node".into();
    }
    if ex.filtered_by_max_age {
        return "filtered_by_max_age: memory is older than max_age".into();
    }
    match (ex.final_rank, ex.exact_rank) {
        (Some(r), _) if r <= k => format!("retrieved: final rank {r} <= top_k {k}"),
        (Some(r), Some(e)) if e <= k => {
            format!("outranked_after_time_decay: exact rank {e} but final rank {r} > top_k {k}")
        }
        (None, Some(e)) if e <= k => {
            format!("hnsw_miss: exact rank {e} but approximate search did not reach it; raise ef")
        }
        (_, Some(e)) => format!("low_similarity: {} memories are more similar", e - 1),
        _ => "unknown".into(),
    }
}
