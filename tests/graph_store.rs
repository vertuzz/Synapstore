//! Knowledge-graph behaviour through the store API.

mod common;

use synapstore::graph::Direction;
use synapstore::store::*;
use synapstore::{Config, Error};

fn f(s: &str, p: &str, o: &str) -> FactInput {
    FactInput { subject: s.into(), predicate: p.into(), object: o.into() }
}

fn names(out: &GraphOutcome) -> Vec<String> {
    let mut v: Vec<String> =
        out.facts.iter().map(|x| format!("{}-{}-{}@{}", x.subject, x.predicate, x.object, x.depth)).collect();
    v.sort();
    v
}

fn family() -> (common::TempDir, SynapStore) {
    let dir = common::TempDir::new("graph");
    let s = SynapStore::open(dir.file("g.redb"), Config::default()).unwrap();
    let facts = vec![
        f("Alice", "works_at", "Acme"),
        f("Bob", "works_at", "Acme"),
        f("Acme", "located_in", "Berlin"),
        f("Alice", "parent_of", "Carol"),
        f("Carol", "studies_at", "TU Berlin"),
        f("Berlin", "capital_of", "Germany"),
    ];
    assert_eq!(s.add_facts(facts.clone(), None, Some(1)).unwrap(), 6);
    assert_eq!(s.add_facts(facts, None, Some(2)).unwrap(), 0, "re-asserting is an upsert");
    (dir, s)
}

#[test]
fn bfs_depth_and_direction() {
    let (_d, s) = family();
    let mut q = GraphQuery::new("alice");
    q.depth = 1;
    let out = s.graph_query(&q).unwrap();
    assert!(out.found);
    assert_eq!(out.subject, "Alice");
    assert_eq!(names(&out), vec!["Alice-parent_of-Carol@1", "Alice-works_at-Acme@1"]);
    q.depth = 2;
    assert_eq!(
        names(&s.graph_query(&q).unwrap()),
        vec![
            "Acme-located_in-Berlin@2",
            "Alice-parent_of-Carol@1",
            "Alice-works_at-Acme@1",
            "Bob-works_at-Acme@2",
            "Carol-studies_at-TU Berlin@2",
        ]
    );
    q.direction = Direction::Out;
    let out = s.graph_query(&q).unwrap();
    assert!(!names(&out).iter().any(|n| n.starts_with("Bob")));
    let mut q = GraphQuery::new("Acme");
    q.direction = Direction::In;
    q.depth = 1;
    assert_eq!(names(&s.graph_query(&q).unwrap()), vec!["Alice-works_at-Acme@1", "Bob-works_at-Acme@1"]);
    q.direction = Direction::Both;
    q.predicate = Some("located_in".into());
    assert_eq!(names(&s.graph_query(&q).unwrap()), vec!["Acme-located_in-Berlin@1"]);
    let missing = s.graph_query(&GraphQuery::new("Nobody")).unwrap();
    assert!(!missing.found && missing.facts.is_empty());
}

#[test]
fn limit_truncates() {
    let (_d, s) = family();
    let mut q = GraphQuery::new("Alice");
    q.depth = 3;
    q.limit = 2;
    let out = s.graph_query(&q).unwrap();
    assert_eq!(out.facts.len(), 2);
    assert!(out.truncated);
}

#[test]
fn pattern_matching_uses_every_index() {
    let (_d, s) = family();
    let m = |a, b, c| {
        let mut v: Vec<String> = s.graph_match(a, b, c, 100).unwrap().into_iter().map(|x| x.subject).collect();
        v.sort();
        v
    };
    assert_eq!(m(None, Some("works_at"), Some("Acme")), vec!["Alice", "Bob"]); // POS
    assert_eq!(m(Some("Alice"), Some("works_at"), None), vec!["Alice"]); // SPO
    assert_eq!(m(Some("Alice"), None, None).len(), 2); // SPO prefix
    assert_eq!(m(None, None, Some("Acme")), vec!["Alice", "Bob"]); // OPS prefix
    assert_eq!(m(None, Some("works_at"), None), vec!["Alice", "Bob"]); // POS prefix
    assert_eq!(m(None, None, None).len(), 6);
    assert_eq!(m(Some("Alice"), None, Some("Carol")), vec!["Alice"]);
    assert!(m(Some("Zed"), None, None).is_empty());
    assert_eq!(s.graph_match(None, None, None, 3).unwrap().len(), 3);
}

#[test]
fn retract_removes_from_all_indices() {
    let (_d, s) = family();
    assert!(s.retract_fact(f("Bob", "works_at", "Acme")).unwrap());
    assert!(!s.retract_fact(f("Bob", "works_at", "Acme")).unwrap());
    assert!(s.graph_match(None, Some("works_at"), Some("Acme"), 10).unwrap().iter().all(|x| x.subject != "Bob"));
    assert!(s.graph_match(Some("Bob"), None, None, 10).unwrap().is_empty());
    let mut q = GraphQuery::new("Acme");
    q.depth = 1;
    assert!(!names(&s.graph_query(&q).unwrap()).iter().any(|n| n.contains("Bob")));
}

#[test]
fn memorize_links_entities_and_facts_to_memories() {
    let (_d, s) = family();
    let mut m = MemorizeInput::text("Alice presented the Q3 roadmap at Acme.");
    m.facts = vec![f("Alice", "presented", "Q3 roadmap")];
    let o = s.memorize(m).unwrap();
    assert_eq!(o.facts_added, 1);
    let out = s.graph_query(&GraphQuery::new("Alice")).unwrap();
    assert_eq!(out.memories.len(), 1);
    assert_eq!(out.memories[0].id, o.chunk_ids[0]);
    let fact = out.facts.iter().find(|x| x.predicate == "presented").unwrap();
    assert_eq!(fact.source_chunk, Some(o.chunk_ids[0]));
    let e = s.add_facts(vec![f("", "x", "y")], None, None).unwrap_err();
    assert!(matches!(e, Error::InvalidInput(_)));
}
