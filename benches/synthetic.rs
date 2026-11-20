//! Iteration 6 benchmark: a synthetic agent life of hundreds of thousands
//! of records (linked facts, unlinked thoughts, episodic logs over months).
//! Reports load throughput, recall quality/latency, graph correctness and
//! latency, time-decay probes, zero-copy allocation, RSS and file size.
//! Knobs: SYNAPSTORE_BENCH_RECORDS (default 200000), SYNAPSTORE_BENCH_DIM (128).

mod support;

use redb::{ReadableDatabase, ReadableTable};
use serde_json::json;
use std::time::Instant;
use synapstore::archive::view;
use synapstore::model::Chunk;
use synapstore::store::{EventQuery, RecallQuery};
use synapstore::synth::{self, Spec, check};
use synapstore::tables::CHUNKS;
use synapstore::{Config, SynapStore};

#[global_allocator]
static ALLOC: support::Counting = support::Counting;

/// Bytes allocated per zero-copy chunk read (must be 0).
fn zero_copy_alloc_per_read(path: &std::path::Path, ids: &[u64]) -> f64 {
    let db = redb::Database::open(path).unwrap();
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(CHUNKS).unwrap();
    let scan = || {
        let mut n = 0usize;
        for id in ids {
            if let Some(g) = t.get(*id).unwrap() {
                n += view::<Chunk>(g.value()).unwrap().text.len();
            }
        }
        n
    };
    scan();
    let (_, bytes) = support::allocated(scan);
    bytes as f64 / ids.len().max(1) as f64
}

fn main() {
    let records = support::env_usize("SYNAPSTORE_BENCH_RECORDS", 200_000);
    let dim = support::env_usize("SYNAPSTORE_BENCH_DIM", 128);
    let scratch = support::Scratch::new("synthetic");
    let path = scratch.0.join("synthetic.redb");
    let store = SynapStore::open(&path, Config::default()).unwrap();
    let spec = Spec::scaled(records, 2026, dim);
    let (world, load) = synth::load(&store, &spec).unwrap();
    eprintln!("loaded: {}", serde_json::to_string(&load).unwrap());

    let recall = check::check_recall(&store, &world, 300, 10, None, 11).unwrap();
    let graph = check::check_graph(&store, &world, 60, 2, 12).unwrap();
    let decay = check::check_decay(&store, &world, &spec).unwrap();

    // Production-shaped recall: default decay, 30-day window.
    let mut lat = Vec::new();
    for (i, m) in world.memories.iter().step_by((world.memories.len() / 200).max(1)).enumerate() {
        let q = RecallQuery {
            vector: Some(m.vector.clone()),
            top_k: 5,
            now_ms: Some(spec.now_ms),
            max_age_ms: (i % 2 == 0).then_some(30 * synapstore::decay::DAY_MS),
            ..Default::default()
        };
        let t = Instant::now();
        store.recall(&q).unwrap();
        lat.push(t.elapsed().as_micros() as u64);
    }
    let (decay_p50, decay_p95) = check::percentiles(&mut lat);
    let mut ev_lat = Vec::new();
    for i in 0..100u64 {
        let t = Instant::now();
        store
            .recent_events(&EventQuery {
                limit: 50,
                since_ms: Some(spec.now_ms - (i + 1) * 86_400_000),
                ..Default::default()
            })
            .unwrap();
        ev_lat.push(t.elapsed().as_micros() as u64);
    }
    let (ev_p50, ev_p95) = check::percentiles(&mut ev_lat);
    let st = store.stats(true).unwrap();
    let ids: Vec<u64> = world.memories.iter().take(5000).map(|m| synapstore::hash::chunk_id(&m.text)).collect();
    drop(store);
    let zc = zero_copy_alloc_per_read(&path, &ids);
    let total_records = st["counts"]["chunks"].as_u64().unwrap_or(0)
        + st["counts"]["episodes"].as_u64().unwrap_or(0)
        + st["counts"]["facts"].as_u64().unwrap_or(0)
        + st["counts"]["terms"].as_u64().unwrap_or(0);
    support::write_result(
        "synthetic",
        &json!({
            "records_requested": records,
            "records_stored": total_records,
            "dim": dim,
            "load": load,
            "recall_at_10": recall.recall_at_k,
            "recall_p50_us": recall.p50_us,
            "recall_p95_us": recall.p95_us,
            "recall_mean_visited": recall.mean_visited,
            "recall_decay_p50_us": decay_p50,
            "recall_decay_p95_us": decay_p95,
            "graph_queries": graph.queries,
            "graph_mismatches": graph.mismatches,
            "graph_edges_checked": graph.edges_checked,
            "graph_p50_us": graph.p50_us,
            "graph_p95_us": graph.p95_us,
            "decay_probes": decay.probes,
            "decay_pass_rate": decay.passed as f64 / decay.probes.max(1) as f64,
            "events_p50_us": ev_p50,
            "events_p95_us": ev_p95,
            "zero_copy_alloc_bytes_per_read": zc,
            "unreachable_nodes": st["index"]["nodes"].as_u64().unwrap_or(0) - st["index"]["reachable_from_entry"].as_u64().unwrap_or(0),
            "counts": st["counts"],
            "index": st["index"],
            "file_mb": st["file_bytes"].as_f64().unwrap_or(0.0) / 1048576.0,
            "peak_rss_mb": support::peak_rss_mb(),
        }),
    );
}
