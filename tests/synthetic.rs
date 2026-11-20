//! Synthetic agent-life dataset at test scale (the full-size run lives in
//! `benches/synthetic.rs`). `SYNAPSTORE_SYNTH_RECORDS` scales it up.

mod common;

use synapstore::synth::{self, Spec, check};
use synapstore::{Config, SynapStore};

#[test]
fn synthetic_world_passes_all_checks() {
    let records = std::env::var("SYNAPSTORE_SYNTH_RECORDS").ok().and_then(|v| v.parse().ok()).unwrap_or(6_000);
    let dir = common::TempDir::new("synth");
    let store = SynapStore::open(dir.file("s.redb"), Config::default()).unwrap();
    let spec = Spec::scaled(records, 7, 64);
    let (world, load) = synth::load(&store, &spec).unwrap();
    println!("{}", serde_json::to_string(&load).unwrap());

    let st = store.stats(true).unwrap();
    assert_eq!(st["counts"]["chunks"], world.memories.len());
    assert_eq!(st["counts"]["episodes"].as_u64().unwrap(), (world.events.len() + world.memories.len()) as u64);
    assert_eq!(st["index"]["reachable_from_entry"], st["index"]["nodes"]);

    let recall = check::check_recall(&store, &world, 100, 10, None, 1).unwrap();
    println!("recall: {}", serde_json::to_string(&recall).unwrap());
    assert!(recall.recall_at_k >= 0.95, "recall@10 = {}", recall.recall_at_k);

    let graph = check::check_graph(&store, &world, 30, 2, 2).unwrap();
    println!("graph: {}", serde_json::to_string(&graph).unwrap());
    assert_eq!(graph.mismatches, 0);
    assert!(graph.edges_checked > 30);

    let decay = check::check_decay(&store, &world, &spec).unwrap();
    assert_eq!(decay.passed, decay.probes, "time-decay probes");
}
