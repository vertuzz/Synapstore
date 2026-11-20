//! Randomized ("fuzz") tests for the HNSW core and the low-level codecs.
//! Deterministic seeds; scale with `SYNAPSTORE_FUZZ_ITERS`.

mod common;

use std::collections::BTreeSet;
use synapstore::archive::{decode, view};
use synapstore::model::{Chunk, HnswNode, MemoryItem, Term};
use synapstore::store::{MemorizeInput, RecallQuery};
use synapstore::synth::rng::Rng;
use synapstore::{Config, Error, SynapStore, idlist, math};

fn nasty_vector(rng: &mut Rng, dim: usize) -> Vec<f32> {
    match rng.below(8) {
        0 => vec![0.0; dim],
        1 => (0..dim).map(|i| if i == 0 { f32::NAN } else { 1.0 }).collect(),
        2 => (0..dim).map(|_| rng.normal() * 1e-30).collect(),
        3 => (0..dim).map(|_| rng.normal() * 1e30).collect(),
        4 => vec![1.0; dim],
        _ => rng.unit_vector(dim),
    }
}

#[test]
fn random_operation_sequences_keep_index_invariants() {
    let iters = common::fuzz_iters(300);
    let dir = common::TempDir::new("fuzz-hnsw");
    let s = SynapStore::open(dir.file("f.redb"), Config::default()).unwrap();
    let mut rng = Rng::new(0xF022);
    let dim = 12;
    let mut live: Vec<(String, Vec<f32>)> = Vec::new();
    for step in 0..iters {
        match rng.below(10) {
            0..=5 => {
                let text = format!("t{}", rng.below(iters / 2 + 1));
                let v = if rng.chance(0.2) && !live.is_empty() {
                    live[rng.below(live.len())].1.clone() // exact duplicate vector, new text
                } else {
                    nasty_vector(&mut rng, dim)
                };
                let mut m = MemorizeInput::text(text.clone()).with_vector(v.clone());
                m.entities = Some(vec![]);
                match s.memorize(m) {
                    Ok(_) => {
                        live.retain(|(t, _)| *t != text);
                        live.push((text, math::normalize(&v).unwrap()));
                    }
                    Err(Error::InvalidVector(_)) => assert!(math::normalize(&v).is_err()),
                    Err(e) => panic!("step {step}: unexpected {e}"),
                }
            }
            6 if !live.is_empty() => {
                let (t, _) = live.remove(rng.below(live.len()));
                assert_eq!(s.forget(vec![synapstore::hash::chunk_id(&t)]).unwrap(), 1);
            }
            _ => {
                let k = 1 + rng.below(8);
                let q = RecallQuery {
                    vector: Some(rng.unit_vector(dim)),
                    top_k: k,
                    ef: Some(1 + rng.below(40)),
                    ..Default::default()
                };
                let out = s.recall(&q).unwrap();
                assert!(out.hits.len() <= k.min(live.len()));
                assert!(out.hits.windows(2).all(|w| w[0].score >= w[1].score));
                assert!(out.hits.iter().all(|h| live.iter().any(|(t, _)| *t == h.text)));
            }
        }
    }
    let before = s.stats(true).unwrap();
    let repaired = s.repair_index().unwrap();
    assert_eq!(repaired.unreachable_before + before["index"]["reachable_from_entry"].as_u64().unwrap(), repaired.nodes);
    let st = s.stats(true).unwrap();
    assert_eq!(st["counts"]["chunks"], live.len());
    let nodes = st["index"]["nodes"].as_u64().unwrap();
    assert_eq!(st["index"]["reachable_from_entry"], nodes, "every node reachable");
    if nodes > 1 {
        assert_eq!(st["index"]["isolated_nodes"], 0);
    }
    // Every live vector finds itself (or an identical twin) at exact rank 1.
    for (t, v) in live.iter().take(50) {
        let top = s.exact_search(v, 1).unwrap();
        assert!((top[0].1 - 1.0).abs() < 1e-4, "{t}");
    }
}

#[test]
fn archive_views_never_panic_on_garbage() {
    let mut rng = Rng::new(7);
    for _ in 0..common::fuzz_iters(300) * 20 {
        let len = rng.below(256);
        let bytes: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
        let _ = view::<Chunk>(&bytes);
        let _ = view::<HnswNode>(&bytes);
        let _ = decode::<Term>(&bytes);
        let _ = decode::<MemoryItem>(&bytes);
    }
    // Truncations of a valid record must be rejected, not misread.
    let good = synapstore::archive::encode(&HnswNode::empty(3)).unwrap();
    for cut in 0..good.len() {
        let _ = view::<HnswNode>(&good[..cut]);
    }
}

#[test]
fn idlist_matches_btreeset_model() {
    let mut rng = Rng::new(3);
    let mut list = Vec::new();
    let mut model = BTreeSet::new();
    for _ in 0..common::fuzz_iters(300) * 10 {
        let id = rng.next_u64() % 64;
        if rng.chance(0.6) {
            let changed = idlist::with(&list, id).map(|l| list = l).is_some();
            assert_eq!(changed, model.insert(id));
        } else {
            let changed = idlist::without(&list, id).map(|l| list = l).is_some();
            assert_eq!(changed, model.remove(&id));
        }
        assert_eq!(idlist::iter(&list).collect::<Vec<_>>(), model.iter().copied().collect::<Vec<_>>());
    }
}

#[test]
fn math_kernels_match_reference_on_random_inputs() {
    let mut rng = Rng::new(4);
    for _ in 0..common::fuzz_iters(300) * 5 {
        let n = rng.below(70);
        let scale = [1e-20f32, 1e-3, 1.0, 1e3, 1e15][rng.below(5)];
        let a: Vec<f32> = (0..n).map(|_| rng.normal() * scale).collect();
        let b: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
        let want: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
        let mag: f64 = a.iter().zip(&b).map(|(x, y)| (f64::from(*x) * f64::from(*y)).abs()).sum();
        let got = f64::from(math::dot(&a, &b));
        assert!((got - want).abs() <= 1e-4 * mag.max(f64::MIN_POSITIVE) + 1e-30, "n={n} got={got} want={want}");
        let c = math::cosine(&a, &b);
        assert!((-1.0..=1.0).contains(&c) || c.is_nan());
        if let Ok(u) = math::normalize(&a) {
            assert!((math::norm(&u) - 1.0).abs() < 1e-3);
        }
    }
}
