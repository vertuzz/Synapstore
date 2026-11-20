//! Model-based fuzzing of the knowledge graph: random assert/retract
//! sequences are mirrored in a `HashSet`, and every index plus BFS
//! traversal is checked against that model.

mod common;

use std::collections::HashSet;
use synapstore::graph::Direction;
use synapstore::hash::normalize;
use synapstore::store::{FactInput, GraphQuery};
use synapstore::synth::check::expected_bfs;
use synapstore::synth::rng::Rng;
use synapstore::{Config, SynapStore};

type Fact = (String, String, String);

const NODES: &[&str] = &["Ann", "Ben", "Cat", "Dan", "Eve", "Fay", "Gus", "Hal", "Ivy", "Jo"];
const PREDS: &[&str] = &["knows", "likes", "works_with", "parent_of"];

fn key(f: &FactInput) -> Fact {
    (normalize(&f.subject), normalize(&f.predicate), normalize(&f.object))
}

#[test]
fn graph_indices_and_bfs_match_model() {
    let dir = common::TempDir::new("fuzz-graph");
    let s = SynapStore::open(dir.file("g.redb"), Config::default()).unwrap();
    let mut rng = Rng::new(0x6EA9);
    let mut model: HashSet<Fact> = HashSet::new();
    for step in 0..common::fuzz_iters(300) {
        let f = FactInput {
            subject: rng.pick(NODES).to_string(),
            predicate: rng.pick(PREDS).to_string(),
            // Random casing must not create distinct entities.
            object: if rng.chance(0.5) { rng.pick(NODES).to_uppercase() } else { rng.pick(NODES).to_string() },
        };
        if rng.chance(0.7) {
            let created = s.add_facts(vec![f.clone()], None, Some(step as u64)).unwrap();
            assert_eq!(created == 1, model.insert(key(&f)), "step {step}");
        } else {
            assert_eq!(s.retract_fact(f.clone()).unwrap(), model.remove(&key(&f)), "step {step}");
        }
        if step % 10 != 0 {
            continue;
        }
        let all: HashSet<Fact> = s
            .graph_match(None, None, None, 100_000)
            .unwrap()
            .iter()
            .map(|x| (normalize(&x.subject), normalize(&x.predicate), normalize(&x.object)))
            .collect();
        assert_eq!(all, model, "step {step}");
        let p = rng.pick(PREDS);
        let o = rng.pick(NODES);
        let pos: HashSet<String> =
            s.graph_match(None, Some(p), Some(o), 1000).unwrap().into_iter().map(|x| normalize(&x.subject)).collect();
        let want: HashSet<String> =
            model.iter().filter(|f| f.1 == *p && f.2 == normalize(o)).map(|f| f.0.clone()).collect();
        assert_eq!(pos, want);
        let facts: Vec<Fact> = model.iter().cloned().collect();
        let root = rng.pick(NODES);
        let depth = 1 + rng.below(3);
        let mut q = GraphQuery::new(*root);
        q.depth = depth;
        q.direction = Direction::Both;
        q.limit = 100_000;
        let got: HashSet<Fact> = s
            .graph_query(&q)
            .unwrap()
            .facts
            .iter()
            .map(|x| (normalize(&x.subject), normalize(&x.predicate), normalize(&x.object)))
            .collect();
        assert_eq!(got, expected_bfs(&facts, &normalize(root), depth), "bfs {root} depth {depth} step {step}");
    }
}
