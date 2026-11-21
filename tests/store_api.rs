//! Store-level semantics: idempotent upserts, space pinning, decay, filters,
//! forgetting, explain verdicts and persistence across reopen.

mod common;

use synapstore::decay::DAY_MS;
use synapstore::hash::chunk_id;
use synapstore::store::explain::ExplainQuery;
use synapstore::store::*;
use synapstore::synth::rng::Rng;
use synapstore::{Config, Error};

const NOW: u64 = 1_790_000_000_000;

fn open(dir: &common::TempDir) -> SynapStore {
    SynapStore::open(dir.file("m.redb"), Config::default()).unwrap()
}

fn vq(v: Vec<f32>, k: usize) -> RecallQuery {
    RecallQuery { vector: Some(v), top_k: k, now_ms: Some(NOW), ..Default::default() }
}

#[test]
fn memorize_is_idempotent_and_upserts_metadata() {
    let dir = common::TempDir::new("idem");
    let s = open(&dir);
    let v = vec![0.1, 0.2, 0.3, 0.4];
    let mut a = MemorizeInput::text("Alice likes Rust").with_vector(v.clone()).at(NOW - DAY_MS);
    a.tags = vec!["lang".into()];
    let o1 = s.memorize(a.clone()).unwrap();
    assert_eq!((o1.created, o1.updated), (1, 0));
    // Same content, different whitespace/case → same id, merged tags, refreshed time.
    let mut b = MemorizeInput::text("  alice LIKES   rust ").with_vector(v.clone()).at(NOW);
    b.tags = vec!["fav".into(), "LANG".into()];
    let o2 = s.memorize(b).unwrap();
    assert_eq!(o2.chunk_ids, o1.chunk_ids);
    assert_eq!((o2.created, o2.updated, o2.unchanged), (0, 1, 1));
    let c = s.get_chunk(o1.chunk_ids[0]).unwrap().unwrap();
    assert_eq!(c.tags, vec!["lang", "fav"]);
    assert_eq!((c.hits, c.created_at, c.updated_at), (2, NOW - DAY_MS, NOW));
    assert_eq!(c.text, "Alice likes Rust");
    let st = s.stats(true).unwrap();
    assert_eq!(st["counts"]["chunks"], 1);
    assert_eq!(st["index"]["nodes"], 1);
}

#[test]
fn vector_space_is_pinned() {
    let dir = common::TempDir::new("space");
    let s = open(&dir);
    s.memorize(MemorizeInput::text("a").with_vector(vec![1.0, 0.0, 0.0])).unwrap();
    let e = s.memorize(MemorizeInput::text("b").with_vector(vec![1.0, 0.0])).unwrap_err();
    assert!(matches!(e, Error::DimensionMismatch { expected: 3, got: 2 }));
    let e = s.memorize(MemorizeInput::text("c")).unwrap_err();
    assert!(matches!(e, Error::EmbeddingMismatch { .. }));
    let e = s.recall(&RecallQuery { text: Some("a".into()), top_k: 1, ..Default::default() }).unwrap_err();
    assert!(matches!(e, Error::EmbeddingMismatch { .. }));
    let e = s.recall(&vq(vec![1.0, 0.0], 1)).unwrap_err();
    assert!(matches!(e, Error::DimensionMismatch { .. }));
    // A failed write must not leave partial state behind.
    assert_eq!(s.stats(false).unwrap()["counts"]["chunks"], 1);
}

#[test]
fn time_decay_prefers_fresh_unless_exact() {
    let dir = common::TempDir::new("decay");
    let s = open(&dir);
    let mut rng = Rng::new(11);
    let base = rng.unit_vector(64);
    let near_old = rng.perturb(&base, 0.06);
    let near_new = rng.perturb(&base, 0.06);
    s.memorize(MemorizeInput::text("old note").with_vector(near_old).at(NOW - 200 * DAY_MS)).unwrap();
    s.memorize(MemorizeInput::text("new note").with_vector(near_new).at(NOW - DAY_MS)).unwrap();
    let hits = s.recall(&vq(base.clone(), 2)).unwrap().hits;
    assert_eq!(hits[0].text, "new note");
    assert!(hits[0].decay_factor > 1.0 && hits[1].decay_factor < 0.6);
    // An exact (sim = 1) old memory is protected from the penalty.
    s.memorize(MemorizeInput::text("exact old").with_vector(base.clone()).at(NOW - 400 * DAY_MS)).unwrap();
    let hits = s.recall(&vq(base.clone(), 3)).unwrap().hits;
    assert_eq!(hits[0].text, "exact old");
    // max_age drops old memories entirely.
    let mut q = vq(base, 3);
    q.max_age_ms = Some(30 * DAY_MS);
    let hits = s.recall(&q).unwrap().hits;
    assert_eq!(hits.iter().map(|h| h.text.as_str()).collect::<Vec<_>>(), vec!["new note"]);
}

#[test]
fn tags_filter_and_adaptive_beam() {
    let dir = common::TempDir::new("tags");
    let s = open(&dir);
    let mut rng = Rng::new(5);
    let inputs: Vec<MemorizeInput> = (0..400)
        .map(|i| {
            let mut m = MemorizeInput::text(format!("memory {i}")).with_vector(rng.unit_vector(32)).at(NOW);
            m.tags = vec![if i % 100 == 0 { "rare".into() } else { "common".into() }];
            m.entities = Some(vec![]);
            m
        })
        .collect();
    assert_eq!(s.memorize_batch(&inputs).unwrap().len(), 400);
    let mut q = vq(rng.unit_vector(32), 4);
    q.tags = vec!["RARE".into()];
    q.ef = Some(8);
    let out = s.recall(&q).unwrap();
    assert_eq!(out.hits.len(), 4, "beam must widen until the filter is satisfied");
    assert!(out.hits.iter().all(|h| h.tags == vec!["rare"]));
}

#[test]
fn forget_tombstones_and_explain() {
    let dir = common::TempDir::new("forget");
    let s = open(&dir);
    let mut rng = Rng::new(9);
    let vecs: Vec<Vec<f32>> = (0..50).map(|_| rng.unit_vector(16)).collect();
    for (i, v) in vecs.iter().enumerate() {
        s.memorize(MemorizeInput::text(format!("item {i}")).with_vector(v.clone()).at(NOW)).unwrap();
    }
    let target = chunk_id("item 7");
    let ex = s
        .explain(&ExplainQuery {
            target,
            vector: Some(vecs[7].clone()),
            top_k: 3,
            now_ms: Some(NOW),
            ..Default::default()
        })
        .unwrap();
    assert!(ex.verdict.starts_with("retrieved"), "{}", ex.verdict);
    assert_eq!((ex.exact_rank, ex.final_rank), (Some(1), Some(1)));
    let ex = s
        .explain(&ExplainQuery {
            target,
            vector: Some(vecs[8].clone()),
            top_k: 1,
            now_ms: Some(NOW),
            ..Default::default()
        })
        .unwrap();
    assert!(ex.verdict.starts_with("low_similarity"), "{}", ex.verdict);

    assert_eq!(s.forget(vec![target, 12345]).unwrap().forgotten, 1);
    let hits = s.recall(&vq(vecs[7].clone(), 3)).unwrap().hits;
    assert!(hits.iter().all(|h| h.id != target));
    assert_eq!(hits.len(), 3, "tombstones still route but are never returned");
    let ex =
        s.explain(&ExplainQuery { target, vector: Some(vecs[7].clone()), top_k: 3, ..Default::default() }).unwrap();
    assert!(ex.verdict.starts_with("not_stored"));
    assert_eq!(s.stats(false).unwrap()["index"]["tombstones"], 1);
    assert!(s.exact_search(&vecs[7], 1).unwrap()[0].0 != target);
    // Re-memorizing revives it in place.
    s.memorize(MemorizeInput::text("item 7").with_vector(vecs[7].clone()).at(NOW)).unwrap();
    assert_eq!(s.recall(&vq(vecs[7].clone(), 1)).unwrap().hits[0].id, target);
}

#[test]
fn builtin_embedder_chunking_and_persistence() {
    let dir = common::TempDir::new("persist");
    let long = "Rust ownership prevents data races at compile time. ".repeat(60);
    {
        let s = open(&dir);
        let o = s.memorize(MemorizeInput::text(long.clone())).unwrap();
        assert!(o.chunk_ids.len() > 1);
        s.memorize(MemorizeInput::text("Grace Hopper popularized the term debugging.")).unwrap();
        s.log_events(&[EventInput { kind: "thought".into(), text: "remember to test".into(), ..Default::default() }])
            .unwrap();
    }
    let s = open(&dir);
    let hits = s
        .recall(&RecallQuery { text: Some("who popularized debugging?".into()), top_k: 1, ..Default::default() })
        .unwrap()
        .hits;
    assert!(hits[0].text.contains("Grace Hopper"));
    assert_eq!(hits[0].entities, vec!["Grace Hopper"]);
    let ev = s.recent_events(&EventQuery { limit: 10, kind: Some("thought".into()), ..Default::default() }).unwrap();
    assert_eq!(ev.len(), 1);
    assert_eq!(ev[0].text, "remember to test");
    let all = s.recent_events(&EventQuery { limit: 100, ..Default::default() }).unwrap();
    assert_eq!(all.len(), 3);
    assert!(all.windows(2).all(|w| w[0].ts >= w[1].ts));
}

#[test]
fn rejects_invalid_input_without_side_effects() {
    let dir = common::TempDir::new("invalid");
    let s = open(&dir);
    assert!(matches!(s.memorize(MemorizeInput::text("")), Err(Error::InvalidInput(_))));
    assert!(matches!(s.memorize(MemorizeInput::text("x").with_vector(vec![0.0; 4])), Err(Error::InvalidVector(_))));
    assert!(matches!(s.memorize(MemorizeInput::text("x").with_vector(vec![f32::NAN])), Err(Error::InvalidVector(_))));
    assert!(matches!(s.recall(&RecallQuery::default()), Err(Error::InvalidInput(_))));
    let empty = RecallQuery { text: Some("anything".into()), top_k: 3, ..Default::default() };
    assert!(s.recall(&empty).unwrap().hits.is_empty());
    let st = s.stats(false).unwrap();
    assert_eq!(st["counts"]["chunks"], 0);
    assert!(st["metrics"]["counters"]["errors"].as_u64().unwrap() >= 4);
}
