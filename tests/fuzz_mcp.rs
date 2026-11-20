//! Fuzz the MCP / JSON-RPC layer in-process: random bytes, mutated JSON and
//! random tool arguments must never panic and must always yield well-formed
//! JSON-RPC responses for requests.

mod common;

use serde_json::{Value, json};
use synapstore::mcp::Server;
use synapstore::synth::rng::Rng;
use synapstore::{Config, SynapStore};

const TOOLS: &[&str] = &[
    "memorize",
    "recall",
    "graph_query",
    "graph_match",
    "add_facts",
    "retract_fact",
    "log_event",
    "recent_events",
    "forget",
    "explain",
    "stats",
    "repair_index",
    "bogus",
];
const KEYS: &[&str] = &[
    "text",
    "vector",
    "query",
    "top_k",
    "facts",
    "entities",
    "tags",
    "subject",
    "predicate",
    "object",
    "depth",
    "direction",
    "max_age",
    "ids",
    "target_id",
    "target_text",
    "kind",
    "limit",
    "ef",
    "detailed",
];

fn random_value(rng: &mut Rng, depth: usize) -> Value {
    match rng.below(if depth > 2 { 6 } else { 9 }) {
        0 => Value::Null,
        1 => json!(rng.chance(0.5)),
        2 => json!(rng.next_u64() as i64),
        3 => json!(rng.normal() * 1e6),
        4 => json!(["Alice", "Bob", "", "7d", "-1", "zz", "0000000000000001", "both"][rng.below(8)]),
        5 => json!((0..rng.below(5)).map(|_| rng.normal()).collect::<Vec<_>>()),
        6 => json!(["a", "b", "c"]),
        7 => Value::Array((0..rng.below(4)).map(|_| random_value(rng, depth + 1)).collect()),
        _ => {
            let mut m = serde_json::Map::new();
            for _ in 0..rng.below(5) {
                m.insert(rng.pick(KEYS).to_string(), random_value(rng, depth + 1));
            }
            Value::Object(m)
        }
    }
}

fn check_reply(input_had_id: bool, reply: Option<String>) {
    let Some(r) = reply else {
        assert!(!input_had_id, "requests must be answered");
        return;
    };
    let v: Value = serde_json::from_str(&r).expect("reply is valid JSON");
    let check = |m: &Value| {
        assert_eq!(m["jsonrpc"], "2.0");
        assert!(m.get("result").is_some() ^ m.get("error").is_some(), "{m}");
    };
    match &v {
        Value::Array(items) => items.iter().for_each(check),
        m => check(m),
    }
}

#[test]
fn random_tool_calls_never_panic() {
    let dir = common::TempDir::new("fuzz-mcp");
    let store = SynapStore::open(dir.file("m.redb"), Config::default()).unwrap();
    let mut server = Server::new(store);
    let mut rng = Rng::new(0x3C9);
    for i in 0..common::fuzz_iters(300) * 3 {
        let mut args = random_value(&mut rng, 0);
        if rng.chance(0.5)
            && let Value::Object(m) = &mut args
        {
            m.insert("text".into(), json!(format!("note {} about Alice and Bob", rng.below(50))));
        }
        let msg = json!({"jsonrpc": "2.0", "id": i, "method": "tools/call",
            "params": {"name": rng.pick(TOOLS), "arguments": args}});
        check_reply(true, server.handle_line(&msg.to_string()));
    }
}

#[test]
fn malformed_frames_never_panic() {
    let dir = common::TempDir::new("fuzz-mcp2");
    let store = SynapStore::open(dir.file("m.redb"), Config::default()).unwrap();
    let mut server = Server::new(store);
    let mut rng = Rng::new(0xBAD);
    let seeds = [
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
        r#"{"jsonrpc":"2.0","id":"x","method":"initialize","params":{"protocolVersion":"2025-06-18"}}"#,
        r#"[{"jsonrpc":"2.0","id":1,"method":"ping"},{"jsonrpc":"2.0","method":"n"}]"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"recall","arguments":{"query":"x"}}}"#,
    ];
    for _ in 0..common::fuzz_iters(300) * 10 {
        let mut bytes = rng.pick(&seeds).as_bytes().to_vec();
        for _ in 0..1 + rng.below(4) {
            let pos = rng.below(bytes.len());
            match rng.below(3) {
                0 => bytes[pos] = rng.next_u64() as u8,
                1 => {
                    bytes.remove(pos);
                }
                _ => bytes.insert(pos, b"{}[]\":,0a\\"[rng.below(10)]),
            }
        }
        let line = String::from_utf8_lossy(&bytes).replace('\n', " ");
        let reply = server.handle_line(&line);
        if let Some(r) = reply {
            let _: Value = serde_json::from_str(&r).expect("valid JSON reply");
        }
    }
}
