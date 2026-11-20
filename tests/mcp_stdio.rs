//! Black-box MCP tests: JSON lines into the binary's stdin, assertions on
//! stdout. (A Python twin lives in tests/python/test_mcp.py.)

mod common;

use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct Session {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: i64,
}

impl Session {
    fn start(db: &std::path::Path, extra: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_synapstore"))
            .arg("--mcp")
            .arg(db)
            .args(extra)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Session { child, stdin, stdout, next_id: 1 }
    }

    fn send_raw(&mut self, line: &str) -> Option<Value> {
        writeln!(self.stdin, "{line}").unwrap();
        let mut out = String::new();
        self.stdout.read_line(&mut out).unwrap();
        (!out.is_empty()).then(|| serde_json::from_str(&out).unwrap())
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        let v = self
            .send_raw(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string())
            .unwrap();
        assert_eq!(v["id"], id);
        v
    }

    fn tool(&mut self, name: &str, args: Value) -> (bool, Value) {
        let v = self.request("tools/call", json!({"name": name, "arguments": args}));
        let r = &v["result"];
        let text: Value = serde_json::from_str(r["content"][0]["text"].as_str().unwrap()).unwrap_or(Value::Null);
        if r["isError"] == false {
            assert_eq!(text, r["structuredContent"], "text and structured content agree");
        }
        (r["isError"] == true, r["structuredContent"].clone())
    }

    fn finish(mut self) {
        drop(self.stdin);
        assert!(self.child.wait().unwrap().success());
    }
}

#[test]
fn full_protocol_session() {
    let dir = common::TempDir::new("mcp");
    let db = dir.file("mem.redb");
    let mut s = Session::start(&db, &[]);
    let init = s.request(
        "initialize",
        json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}),
    );
    assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(init["result"]["serverInfo"]["name"], "synapstore");
    // Notifications get no reply: the next line read must answer the ping.
    writeln!(s.stdin, "{}", json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).unwrap();
    assert_eq!(s.request("ping", json!({}))["result"], json!({}));
    let tools = s.request("tools/list", json!({}));
    let names: Vec<&str> =
        tools["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    for t in ["memorize", "recall", "graph_query"] {
        assert!(names.contains(&t));
    }

    let v = vec![0.0, 1.0, 0.0, 0.0];
    let (err, m) = s.tool(
        "memorize",
        json!({"text": "Ada Lovelace wrote the first program", "vector": v,
        "facts": [["Ada Lovelace", "collaborated_with", "Charles Babbage"]], "tags": ["history"]}),
    );
    assert!(!err, "{m}");
    assert_eq!(m["created"], 1);
    let id = m["chunk_ids"][0].as_str().unwrap().to_string();
    assert_eq!(id.len(), 16);
    s.tool("memorize", json!({"text": "Unrelated grocery list", "vector": [1, 0, 0, 0]}));

    let (_, r) = s.tool("recall", json!({"vector": [0.1, 0.9, 0.0, 0.0], "top_k": 1, "max_age": "1d"}));
    assert_eq!(r["hits"][0]["id"], id.as_str());
    assert_eq!(r["hits"][0]["tags"], json!(["history"]));
    let (_, g) = s.tool("graph_query", json!({"subject": "Charles Babbage", "depth": 1}));
    assert_eq!(g["facts"][0]["subject"], "Ada Lovelace");
    let (_, gm) = s.tool("graph_match", json!({"predicate": "collaborated_with"}));
    assert_eq!(gm["count"], 1);
    let (_, ev) = s.tool("log_event", json!({"kind": "action", "text": "ran the analysis"}));
    assert!(ev["id"].is_string());
    let (_, re) = s.tool("recent_events", json!({"kind": "action"}));
    assert_eq!(re["events"][0]["text"], "ran the analysis");
    let (_, ex) = s.tool("explain", json!({"target_id": id, "vector": [0.1, 0.9, 0, 0], "top_k": 1}));
    assert!(ex["verdict"].as_str().unwrap().starts_with("retrieved"));
    let (_, st) = s.tool("stats", json!({"detailed": true}));
    assert_eq!(st["counts"]["chunks"], 2);
    let (_, fg) = s.tool("forget", json!({"text": "Unrelated grocery list"}));
    assert_eq!(fg["forgotten"], 1);
    let (_, rf) = s.tool(
        "retract_fact",
        json!({"subject": "Ada Lovelace", "predicate": "collaborated_with", "object": "Charles Babbage"}),
    );
    assert_eq!(rf["retracted"], true);
    s.finish();
    let log = std::fs::read_to_string(dir.file("mem.redb.log")).unwrap();
    assert!(log.contains(" INFO recall ") && log.contains("latency_us="), "{log}");
}

#[test]
fn protocol_errors_are_reported_not_fatal() {
    let dir = common::TempDir::new("mcp-err");
    let mut s = Session::start(&dir.file("e.redb"), &["--no-log"]);
    assert_eq!(s.send_raw("{not json").unwrap()["error"]["code"], -32700);
    assert_eq!(s.send_raw("[]").unwrap()["error"]["code"], -32600);
    assert_eq!(s.send_raw(r#"{"jsonrpc":"2.0","id":1}"#).unwrap()["error"]["code"], -32600);
    assert_eq!(s.request("no/such/method", json!({}))["error"]["code"], -32601);
    assert_eq!(s.request("tools/call", json!({"name": "nope"}))["error"]["code"], -32602);
    assert_eq!(s.request("tools/call", json!({}))["error"]["code"], -32602);
    let (err, e) = s.tool("memorize", json!({"text": 42}));
    assert!(err);
    assert_eq!(e["error"]["kind"], "invalid_input");
    let (err, e) = s.tool("recall", json!({"vector": [0, 0, 0]}));
    assert!(err, "{e}");
    let (err, _) = s.tool("recall", json!({"query": "anything"}));
    assert!(!err, "empty store recall is fine");
    let batch = s.send_raw(r#"[{"jsonrpc":"2.0","id":"a","method":"ping"},{"jsonrpc":"2.0","method":"notifications/x"},{"jsonrpc":"2.0","id":"b","method":"ping"}]"#).unwrap();
    assert_eq!(batch.as_array().unwrap().len(), 2);
    let init = s.request("initialize", json!({"protocolVersion": "1999-01-01"}));
    assert_eq!(init["result"]["protocolVersion"], synapstore::mcp::PROTOCOL_VERSIONS[0]);
    s.finish();
}

#[test]
fn cli_stats_and_dump() {
    let dir = common::TempDir::new("cli");
    let db = dir.file("c.redb");
    let mut s = Session::start(&db, &["--no-log"]);
    s.tool("memorize", json!({"text": "Grace Hopper found a moth in the Mark II", "facts": [["Grace Hopper", "worked_on", "Mark II"]]}));
    s.finish();
    let bin = env!("CARGO_BIN_EXE_synapstore");
    let out = Command::new(bin).args(["stats", db.to_str().unwrap(), "--detailed"]).output().unwrap();
    assert!(out.status.success());
    let st: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(st["counts"]["facts"], 1);
    let out = Command::new(bin).args(["dump", db.to_str().unwrap(), "--vectors"]).output().unwrap();
    let lines: Vec<Value> =
        String::from_utf8(out.stdout).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let kinds: std::collections::HashSet<&str> = lines.iter().map(|l| l["type"].as_str().unwrap()).collect();
    for k in ["meta", "term", "chunk", "hnsw_node", "fact", "event"] {
        assert!(kinds.contains(k), "missing {k}");
    }
    assert_eq!(lines.iter().find(|l| l["type"] == "chunk").unwrap()["vector"].as_array().unwrap().len(), 384);
    let bad = Command::new(bin).args(["--bogus"]).output().unwrap();
    assert!(!bad.status.success());
}
