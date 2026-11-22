//! 0.3 behaviour: offline start with a remote embedder, lexical fallback,
//! `reindex` (data kept, split memories joined), `eval` and usage logging.

mod common;

use common::TempDir;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use synapstore::auth::Scope;
use synapstore::embed::EmbedderSpec;
use synapstore::mcp::Server;
use synapstore::store::{EventInput, EventQuery, FactInput, MemorizeInput, RecallQuery};
use synapstore::{Config, SynapStore};

/// OpenAI-style `/v1/embeddings` that embeds text as counts of a, b, c, d.
fn fake_server() -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    std::thread::spawn(move || {
        for s in l.incoming().flatten() {
            let mut r = BufReader::new(s.try_clone().unwrap());
            let (mut line, mut len) = (String::new(), 0usize);
            r.read_line(&mut line).unwrap();
            loop {
                let mut h = String::new();
                r.read_line(&mut h).unwrap();
                if h.trim().is_empty() {
                    break;
                }
                if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; len];
            r.read_exact(&mut body).unwrap();
            let req: Value = serde_json::from_slice(&body).unwrap();
            let data: Vec<Value> = req["input"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let t = t.as_str().unwrap().to_lowercase();
                    let v: Vec<f32> =
                        ['a', 'b', 'c', 'd'].iter().map(|c| t.matches(*c).count() as f32 + 0.01).collect();
                    json!({"index": i, "embedding": v})
                })
                .collect();
            let out = json!({"data": data}).to_string();
            let mut s = s;
            write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{out}", out.len()).unwrap();
        }
    });
    format!("http://{addr}/v1")
}

fn remote(spec: &str) -> Config {
    Config { embedder: EmbedderSpec::parse(spec).unwrap(), ..Config::default() }
}

#[test]
fn recorded_dimension_opens_offline_and_recall_degrades_to_words() {
    let d = TempDir::new("offline");
    let db = d.file("m.redb");
    let s = SynapStore::open(&db, remote(&format!("openai:fake-embed@{}", fake_server()))).unwrap();
    s.memorize(MemorizeInput::text("aaaa aaa alpha")).unwrap();
    s.memorize(MemorizeInput::text("dddd ddd delta")).unwrap();
    drop(s);
    // Same model, endpoint down: the store opens without probing.
    let s = SynapStore::open(&db, remote("openai:fake-embed@http://127.0.0.1:1/v1")).unwrap();
    assert_eq!(s.embedder().source(), "remote:fake-embed:4");
    let out = s.recall(&RecallQuery::text("delta", 2)).unwrap();
    assert_eq!(out.embedding, "lexical-fallback");
    assert!(out.warning.is_some());
    assert_eq!(out.hits[0].text, "dddd ddd delta");
    let err = s.memorize(MemorizeInput::text("new")).unwrap_err();
    assert_eq!(err.kind(), "embedding_unavailable");
}

fn long_memory() -> String {
    (0..30)
        .map(|i| format!("Sentence number {i} about Alice and her work at Acme in Amsterdam."))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn reindex_keeps_data_and_joins_split_memories() {
    let d = TempDir::new("reindex");
    let (src, dst) = (d.file("old.redb"), d.file("new.redb"));
    let s = SynapStore::open(&src, Config::default()).unwrap();
    let mut big = MemorizeInput::text(format!("ВАЖНО: {}", long_memory()));
    big.tags = vec!["status".into()];
    // As extracted by 0.2 heuristics: one junk name.
    big.entities = Some(vec!["ВАЖНО".into(), "Amsterdam".into()]);
    big.facts = vec![FactInput { subject: "Alice".into(), predicate: "works_at".into(), object: "Acme".into() }];
    let big_out = s.memorize(big).unwrap();
    assert!(big_out.chunk_ids.len() > 1, "fixture must be split by the 1200-char chunker");
    s.memorize(MemorizeInput::text("Bob likes green tea.")).unwrap();
    s.log_events(&[EventInput { kind: "note".into(), text: "deployed".into(), timestamp_ms: Some(5), chunk_id: None }])
        .unwrap();
    let (secret, _) = s.create_token("agent", Scope::Read, None).unwrap();
    let events_before = s.recent_events(&EventQuery { limit: 100, ..Default::default() }).unwrap().len();
    drop(s);

    let t = SynapStore::open(&dst, remote(&format!("openai:fake-embed@{}", fake_server()))).unwrap();
    let rep = synapstore::reindex::reindex(&src, &t, true, &|_| {}).unwrap();
    assert_eq!(
        (rep.chunks_in, rep.memories, rep.merged_groups, rep.chunks_out),
        (big_out.chunk_ids.len() + 1, 2, 1, 2)
    );
    assert_eq!((rep.facts, rep.tokens, rep.events), (1, 1, events_before));
    assert!(rep.entities_dropped >= 1, "{rep:?}");

    let hits = t.recall(&RecallQuery::text("Alice Acme Amsterdam", 2)).unwrap().hits;
    let joined = hits.iter().find(|h| h.text.contains("Sentence number 29")).expect("whole memory");
    assert!(joined.text.starts_with("ВАЖНО: Sentence number 0"));
    assert_eq!(joined.tags, vec!["status"]);
    assert!(!joined.entities.iter().any(|e| e == "ВАЖНО"), "{:?}", joined.entities);
    let facts = t.graph_match(Some("Alice"), None, None, 10).unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].source_chunk, Some(joined.id));
    assert!(t.authenticate(&secret).unwrap().is_some(), "tokens survive");
    assert_eq!(t.recent_events(&EventQuery { limit: 100, ..Default::default() }).unwrap().len(), events_before);
    // The source was only read.
    let again = SynapStore::open(&src, Config::default()).unwrap();
    assert_eq!(again.recall(&RecallQuery::text("green tea", 1)).unwrap().hits[0].text, "Bob likes green tea.");
}

#[test]
fn reindex_refuses_a_non_empty_destination() {
    let d = TempDir::new("reindex-busy");
    let (src, dst) = (d.file("a.redb"), d.file("b.redb"));
    SynapStore::open(&src, Config::default()).unwrap().memorize(MemorizeInput::text("x")).unwrap();
    let t = SynapStore::open(&dst, Config::default()).unwrap();
    t.memorize(MemorizeInput::text("already here")).unwrap();
    assert!(synapstore::reindex::reindex(&src, &t, true, &|_| {}).is_err());
}

#[test]
fn eval_scores_snippet_gold() {
    let d = TempDir::new("eval");
    let s = SynapStore::open(d.file("m.redb"), Config::default()).unwrap();
    s.memorize(MemorizeInput::text("The heat pump is a Daikin Altherma.")).unwrap();
    s.memorize(MemorizeInput::text("Our dentist is in Haarlem.")).unwrap();
    let cases = synapstore::eval::parse_cases("heat pump brand\tdaikin\nwho is the dentist\thaarlem\n").unwrap();
    let rep = synapstore::eval::run(&s, &cases, 5).unwrap();
    assert_eq!((rep.queries, rep.at_5), (2, 100.0));
}

#[test]
fn decay_tags_spare_durable_memories() {
    let d = TempDir::new("decay-tags");
    let cfg = Config { decay_tags: vec!["status".into()], ..Config::default() };
    let s = SynapStore::open(d.file("m.redb"), cfg).unwrap();
    let mut old = MemorizeInput::text("Alice is the manager of the Falcon team");
    old.timestamp_ms = Some(1);
    s.memorize(old).unwrap();
    let mut st = MemorizeInput::text("Alice status of the Falcon team today");
    st.tags = vec!["status".into()];
    st.timestamp_ms = Some(1);
    s.memorize(st).unwrap();
    let hits = s.recall(&RecallQuery::text("Alice Falcon team", 2)).unwrap().hits;
    let durable = hits.iter().find(|h| h.tags.is_empty()).unwrap();
    let volatile = hits.iter().find(|h| !h.tags.is_empty()).unwrap();
    assert_eq!(durable.decay_factor, 1.0);
    assert!(volatile.decay_factor < 0.6, "{}", volatile.decay_factor);
}

#[test]
fn usage_log_records_clients_and_recalls() {
    let d = TempDir::new("usage");
    let s = SynapStore::open(d.file("m.redb"), Config::default()).unwrap();
    s.memorize(MemorizeInput::text("Bob likes green tea.")).unwrap();
    let mut srv = Server::scoped(s.clone(), Scope::Read).with_client("judge", true);
    let init = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": {"name": "codex"}}});
    srv.handle_json(init).unwrap();
    let call = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "recall", "arguments": {"query": "what tea", "top_k": 1}}});
    srv.handle_json(call).unwrap();
    let mut u = Value::Null;
    for _ in 0..100 {
        u = s.usage(5).unwrap();
        if u["calls"] == 1 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(u["by_client"]["judge/codex"]["recall"], 1, "{u}");
    assert_eq!(u["most_recalled"][0]["recalled"], 1);
    assert_eq!(u["recall"]["weakest"][0]["query"], "what tea");
}
