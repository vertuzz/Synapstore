//! The OpenAI-compatible embedder against a fake `/v1/embeddings` server.

use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use synapstore::embed::EmbedderSpec;
use synapstore::embed::remote::RemoteConfig;
use synapstore::store::{MemorizeInput, RecallQuery};
use synapstore::{Config, SynapStore};

/// Embeds text as a bag of its first letters (4 dims), OpenAI-style.
fn fake_server() -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    std::thread::spawn(move || {
        for s in l.incoming().flatten() {
            let mut r = BufReader::new(s.try_clone().unwrap());
            let (mut line, mut len, mut auth) = (String::new(), 0usize, String::new());
            r.read_line(&mut line).unwrap();
            loop {
                let mut h = String::new();
                r.read_line(&mut h).unwrap();
                if h.trim().is_empty() {
                    break;
                }
                let lower = h.to_ascii_lowercase();
                if let Some(v) = lower.strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap();
                }
                if lower.starts_with("authorization:") {
                    auth = h.trim().to_string();
                }
            }
            let mut body = vec![0; len];
            r.read_exact(&mut body).unwrap();
            let req: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(req["model"], "fake-embed");
            let data: Vec<Value> = req["input"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let t = t.as_str().unwrap().to_lowercase();
                    let v: Vec<f32> =
                        ['a', 'b', 'c', 'd'].iter().map(|c| t.matches(*c).count() as f32 + 0.01).collect();
                    json!({"object": "embedding", "index": i, "embedding": v})
                })
                .rev()
                .collect();
            let out = json!({"object": "list", "data": data, "model": "fake-embed", "auth_seen": auth}).to_string();
            let mut s = s;
            write!(
                s,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{out}",
                out.len()
            )
            .unwrap();
        }
    });
    format!("http://{addr}/v1")
}

#[test]
fn remote_embedder_drives_a_store() {
    let base = fake_server();
    let dir = std::env::temp_dir().join(format!("synapstore-remote-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("m.redb");
    std::fs::remove_file(&db).ok();
    let spec = EmbedderSpec::Remote(RemoteConfig::parse(&format!("fake-embed@{base}")).unwrap());
    let s = SynapStore::open(&db, Config { embedder: spec, ..Config::default() }).unwrap();
    assert_eq!((s.embedder().dim(), s.embedder().source()), (4, "remote:fake-embed:4"));
    s.memorize(MemorizeInput::text("aaaa aaa")).unwrap();
    s.memorize(MemorizeInput::text("dddd ddd")).unwrap();
    let hits = s.recall(&RecallQuery::text("aaaaa", 1)).unwrap().hits;
    assert_eq!(hits[0].text, "aaaa aaa");
    drop(s);
    // Reopening without --embedder reconnects to the same endpoint.
    let s = SynapStore::open(&db, Config::default()).unwrap();
    assert_eq!(s.embedder().source(), "remote:fake-embed:4");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn unreachable_endpoint_fails_at_open() {
    let spec = EmbedderSpec::Remote(RemoteConfig::parse("m@http://127.0.0.1:1/v1").unwrap());
    let dir = std::env::temp_dir().join(format!("synapstore-remote-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let err = SynapStore::open(dir.join("m.redb"), Config { embedder: spec, ..Config::default() }).err().unwrap();
    assert!(err.to_string().contains("embedding endpoint"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}
