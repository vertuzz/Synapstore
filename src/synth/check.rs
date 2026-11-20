//! Ground-truth verification of a loaded synthetic world.

use super::Spec;
use super::rng::Rng;
use super::world::{World, noise_for_cos};
use crate::decay::DecayParams;
use crate::error::Result;
use crate::hash::normalize;
use crate::store::{GraphQuery, RecallQuery, SynapStore};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// `(p50, p95)` of microsecond samples.
pub fn percentiles(samples: &mut [u64]) -> (u64, u64) {
    if samples.is_empty() {
        return (0, 0);
    }
    samples.sort_unstable();
    let at = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
    (at(0.50), at(0.95))
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RecallCheck {
    pub queries: usize,
    pub k: usize,
    pub recall_at_k: f64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub mean_visited: f64,
}

/// HNSW recall@k against brute force (decay disabled to isolate the index).
pub fn check_recall(
    store: &SynapStore,
    w: &World,
    queries: usize,
    k: usize,
    ef: Option<usize>,
    seed: u64,
) -> Result<RecallCheck> {
    let mut rng = Rng::new(seed);
    let pool: Vec<usize> = (0..w.memories.len()).filter(|i| w.memories[*i].topic != usize::MAX).collect();
    let dim = w.memories.first().map_or(1, |m| m.vector.len());
    let (mut hit, mut total, mut visited) = (0usize, 0usize, 0u64);
    let mut lat = Vec::with_capacity(queries);
    let decay = DecayParams { enabled: false, ..DecayParams::default() };
    for _ in 0..queries {
        let base = &w.memories[*rng.pick(&pool)].vector;
        let q = rng.perturb(base, noise_for_cos(0.9, dim));
        let exact: HashSet<u64> = store.exact_search(&q, k)?.into_iter().map(|(id, _)| id).collect();
        let started = Instant::now();
        let out =
            store.recall(&RecallQuery { vector: Some(q), top_k: k, ef, decay: Some(decay), ..Default::default() })?;
        lat.push(started.elapsed().as_micros() as u64);
        visited += out.visited;
        hit += out.hits.iter().filter(|h| exact.contains(&h.id)).count();
        total += exact.len();
    }
    let (p50_us, p95_us) = percentiles(&mut lat);
    Ok(RecallCheck {
        queries,
        k,
        recall_at_k: if total == 0 { 1.0 } else { hit as f64 / total as f64 },
        p50_us,
        p95_us,
        mean_visited: visited as f64 / queries.max(1) as f64,
    })
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct GraphCheck {
    pub queries: usize,
    pub mismatches: usize,
    pub edges_checked: usize,
    pub p50_us: u64,
    pub p95_us: u64,
}

type Fact = (String, String, String);

/// Expected depth-limited BFS (both directions) over the generator's facts.
pub fn expected_bfs(facts: &[Fact], root: &str, depth: usize) -> HashSet<Fact> {
    let mut adj: HashMap<&str, Vec<&Fact>> = HashMap::new();
    for f in facts {
        adj.entry(f.0.as_str()).or_default().push(f);
        adj.entry(f.2.as_str()).or_default().push(f);
    }
    let mut seen_nodes: HashSet<&str> = HashSet::from([root]);
    let mut frontier = vec![root];
    let mut out = HashSet::new();
    for _ in 0..depth {
        let mut next = Vec::new();
        for n in frontier {
            for f in adj.get(n).into_iter().flatten() {
                out.insert((*f).clone());
                let other = if f.0 == n { f.2.as_str() } else { f.0.as_str() };
                if seen_nodes.insert(other) {
                    next.push(other);
                }
            }
        }
        frontier = next;
    }
    out
}

/// Compare `graph_query(depth)` with the expected traversal for sampled people.
pub fn check_graph(store: &SynapStore, w: &World, samples: usize, depth: usize, seed: u64) -> Result<GraphCheck> {
    let canon = |s: &str| normalize(s);
    let mut facts: Vec<Fact> =
        w.facts.iter().map(|f| (canon(&f.subject), canon(&f.predicate), canon(&f.object))).collect();
    facts.sort();
    facts.dedup();
    let mut rng = Rng::new(seed);
    let mut res = GraphCheck { queries: samples, ..Default::default() };
    let mut lat = Vec::with_capacity(samples);
    for _ in 0..samples {
        let who = rng.pick(&w.people).clone();
        let mut q = GraphQuery::new(&who);
        q.depth = depth;
        q.limit = 1_000_000;
        q.memories = 0;
        let started = Instant::now();
        let out = store.graph_query(&q)?;
        lat.push(started.elapsed().as_micros() as u64);
        let got: HashSet<Fact> =
            out.facts.iter().map(|f| (canon(&f.subject), canon(&f.predicate), canon(&f.object))).collect();
        let want = expected_bfs(&facts, &canon(&who), depth);
        res.edges_checked += want.len();
        if got != want {
            res.mismatches += 1;
        }
    }
    (res.p50_us, res.p95_us) = percentiles(&mut lat);
    Ok(res)
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DecayCheck {
    pub probes: usize,
    pub passed: usize,
}

/// Each probe's expected memory must rank first under default decay, ahead
/// of its counterpart.
pub fn check_decay(store: &SynapStore, w: &World, spec: &Spec) -> Result<DecayCheck> {
    let mut res = DecayCheck { probes: w.probes.len(), passed: 0 };
    for p in &w.probes {
        let out = store.recall(&RecallQuery {
            vector: Some(p.query.clone()),
            top_k: 2,
            now_ms: Some(spec.now_ms),
            tags: vec!["probe".into()],
            ..Default::default()
        })?;
        let texts: Vec<&str> = out.hits.iter().map(|h| h.text.as_str()).collect();
        if texts.first() == Some(&p.expect_text.as_str()) {
            res.passed += 1;
        }
    }
    Ok(res)
}
