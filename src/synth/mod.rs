//! Synthetic benchmark dataset (PRD §8): load a generated world into a store
//! and verify recall, graph traversal and time-decay against ground truth.

pub mod check;
pub mod rng;
pub mod world;

use crate::decay::DAY_MS;
use crate::error::Result;
use crate::store::{EventInput, MemorizeInput, SynapStore};
use serde::Serialize;
use std::time::Instant;

/// Fixed simulated "now" so runs are reproducible (2026-09-21).
pub const SIM_NOW_MS: u64 = 1_790_000_000_000;

#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
    pub seed: u64,
    pub dim: usize,
    pub chunks: usize,
    pub events: usize,
    pub span_days: u64,
    pub now_ms: u64,
    pub batch: usize,
}

impl Spec {
    /// Split a total record budget: 1/3 semantic memories, 2/3 episodic events
    /// (facts and entities come on top, derived from the memories).
    pub fn scaled(records: usize, seed: u64, dim: usize) -> Spec {
        Spec {
            seed,
            dim: dim.max(8),
            chunks: (records / 3).max(50),
            events: records - records / 3,
            span_days: 180,
            now_ms: SIM_NOW_MS,
            batch: 500,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LoadReport {
    pub memories: usize,
    pub events: usize,
    pub facts: usize,
    pub load_secs: f64,
    pub memories_per_sec: f64,
    pub events_per_sec: f64,
}

/// Generate the world for `spec` and memorize it into `store`.
pub fn load(store: &SynapStore, spec: &Spec) -> Result<(world::World, LoadReport)> {
    let w = world::generate(spec.seed, spec.dim, spec.chunks, spec.events, spec.now_ms, spec.span_days);
    let started = Instant::now();
    for batch in w.memories.chunks(spec.batch.max(1)) {
        let inputs: Vec<MemorizeInput> = batch
            .iter()
            .map(|m| MemorizeInput {
                text: m.text.clone(),
                vector: Some(m.vector.clone()),
                entities: Some(m.entities.clone()),
                facts: m.facts.clone(),
                tags: m.tags.clone(),
                source: Some("synthetic".into()),
                kind: Some("memory".into()),
                timestamp_ms: Some(m.ts),
                supersedes: Vec::new(),
            })
            .collect();
        store.memorize_batch(&inputs)?;
    }
    let mem_secs = started.elapsed().as_secs_f64();
    store.add_facts(w.facts.clone(), None, Some(spec.now_ms - spec.span_days * DAY_MS))?;
    let ev_start = Instant::now();
    for batch in w.events.chunks(spec.batch.max(1) * 4) {
        let evs: Vec<EventInput> = batch
            .iter()
            .map(|e| EventInput {
                kind: e.kind.clone(),
                text: e.text.clone(),
                timestamp_ms: Some(e.ts),
                chunk_id: None,
            })
            .collect();
        store.log_events(&evs)?;
    }
    let ev_secs = ev_start.elapsed().as_secs_f64();
    let report = LoadReport {
        memories: w.memories.len(),
        events: w.events.len(),
        facts: w.facts.len(),
        load_secs: started.elapsed().as_secs_f64(),
        memories_per_sec: w.memories.len() as f64 / mem_secs.max(1e-9),
        events_per_sec: w.events.len() as f64 / ev_secs.max(1e-9),
    };
    Ok((w, report))
}
