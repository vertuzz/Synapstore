//! Black-box tests of the Streamable HTTP transport and token auth.

use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use synapstore::auth::Scope;
use synapstore::http::{self, ServeOptions};
use synapstore::{Config, SynapStore};

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Value,
}

impl Reply {
    fn header(&self, k: &str) -> Option<&str> {
        self.headers.iter().find(|(h, _)| h.eq_ignore_ascii_case(k)).map(|(_, v)| v.as_str())
    }
}

fn request(addr: SocketAddr, method: &str, headers: &[(&str, &str)], body: &str) -> Reply {
    let mut s = TcpStream::connect(addr).unwrap();
    let mut req =
        format!("{method} /mcp HTTP/1.1\r\nHost: test\r\nContent-Length: {}\r\nConnection: close\r\n", body.len());
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    req.push_str(body);
    s.write_all(req.as_bytes()).unwrap();
    let mut r = BufReader::new(s);
    let mut line = String::new();
    r.read_line(&mut line).unwrap();
    let status = line.split_whitespace().nth(1).unwrap().parse().unwrap();
    let mut headers = Vec::new();
    loop {
        line.clear();
        r.read_line(&mut line).unwrap();
        if line.trim().is_empty() {
            break;
        }
        let (k, v) = line.split_once(':').unwrap();
        headers.push((k.trim().to_string(), v.trim().to_string()));
    }
    let mut body = String::new();
    r.read_to_string(&mut body).unwrap();
    Reply { status, headers, body: serde_json::from_str(&body).unwrap_or(Value::Null) }
}

fn rpc(addr: SocketAddr, token: &str, session: Option<&str>, msg: Value) -> Reply {
    let auth = format!("Bearer {token}");
    let mut h = vec![("Content-Type", "application/json"), ("Authorization", auth.as_str())];
    if let Some(s) = session {
        h.push(("Mcp-Session-Id", s));
    }
    request(addr, "POST", &h, &msg.to_string())
}

fn init() -> Value {
    json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
           "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}})
}

fn call(id: u64, tool: &str, args: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": tool, "arguments": args}})
}

fn start(opts: ServeOptions) -> (SynapStore, SocketAddr, tempdir::Dir) {
    let dir = tempdir::Dir::new();
    let store = SynapStore::open(dir.path().join("m.redb"), Config::default()).unwrap();
    let listener = http::bind(&ServeOptions { listen: "127.0.0.1:0".into(), ..opts.clone() }).unwrap();
    let addr = listener.local_addr().unwrap();
    let s = store.clone();
    std::thread::spawn(move || http::serve(s, opts, listener));
    (store, addr, dir)
}

mod tempdir {
    pub struct Dir(std::path::PathBuf);
    impl Dir {
        pub fn new() -> Self {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let p = std::env::temp_dir().join(format!("synapstore-http-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&p).unwrap();
            Dir(p)
        }
        pub fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }
}

#[test]
fn rejects_missing_and_bad_tokens() {
    let (store, addr, _d) = start(ServeOptions::default());
    let r = request(addr, "POST", &[("Content-Type", "application/json")], "{}");
    assert_eq!(r.status, 401);
    assert!(r.header("www-authenticate").unwrap().starts_with("Bearer realm=\"synapstore\""));
    let fake = format!("syn_{}", "A".repeat(43));
    assert_eq!(rpc(addr, &fake, None, init()).status, 401);
    let (secret, _) = store.create_token("t", Scope::Write, None).unwrap();
    assert_eq!(rpc(addr, &secret, None, init()).status, 200);
    // Secrets are stored hashed: the file never contains them.
    drop(store);
    let raw = std::fs::read(_d.path().join("m.redb")).unwrap();
    assert!(!raw.windows(secret.len()).any(|w| w == secret.as_bytes()));
}

#[test]
fn full_session_flow_with_write_token() {
    let (store, addr, _d) = start(ServeOptions::default());
    let (tok, _) = store.create_token("agent", Scope::Write, Some(30)).unwrap();
    let r = rpc(addr, &tok, None, init());
    assert_eq!(r.body["result"]["protocolVersion"], "2025-06-18");
    let sid = r.header("mcp-session-id").unwrap().to_string();
    let note = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
    assert_eq!(rpc(addr, &tok, Some(&sid), note).status, 202);
    let m = rpc(addr, &tok, Some(&sid), call(2, "memorize", json!({"text": "Alice prefers dark roast coffee"})));
    assert_eq!(m.body["result"]["isError"], false, "{}", m.body);
    let q = rpc(addr, &tok, Some(&sid), call(3, "recall", json!({"query": "what coffee does Alice like"})));
    assert!(q.body["result"]["structuredContent"]["hits"][0]["text"].as_str().unwrap().contains("dark roast"));
    // The session is bound to its token.
    let (other, _) = store.create_token("other", Scope::Write, None).unwrap();
    assert_eq!(rpc(addr, &other, Some(&sid), call(4, "stats", json!({}))).status, 404);
    assert_eq!(rpc(addr, &tok, Some("nope"), call(5, "stats", json!({}))).status, 404);
    // Stateless requests (no session id) also work.
    assert_eq!(rpc(addr, &tok, None, call(6, "stats", json!({}))).status, 200);
    let auth = format!("Bearer {tok}");
    let del = |sid: &str| request(addr, "DELETE", &[("Authorization", &auth), ("Mcp-Session-Id", sid)], "").status;
    assert_eq!(del(&sid), 204);
    assert_eq!(del(&sid), 404);
    assert!(store.list_tokens().unwrap().iter().any(|t| t.name == "agent" && t.last_used_at.is_some()));
}

#[test]
fn read_scope_is_enforced_and_revocation_is_immediate() {
    let (store, addr, _d) = start(ServeOptions::default());
    let (ro, info) = store.create_token("reader", Scope::Read, None).unwrap();
    let list = rpc(addr, &ro, None, json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}));
    let names: Vec<&str> =
        list.body["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"recall") && !names.contains(&"memorize") && !names.contains(&"repair_index"));
    let w = rpc(addr, &ro, None, call(2, "memorize", json!({"text": "should not be stored"})));
    assert_eq!(w.body["result"]["structuredContent"]["error"]["kind"], "forbidden");
    assert_eq!(store.stats(false).unwrap()["counts"]["chunks"], 0);
    assert_eq!(rpc(addr, &ro, None, call(3, "recall", json!({"query": "x"}))).body["result"]["isError"], false);
    assert_eq!(store.revoke_token(&info.id).unwrap().len(), 1);
    assert_eq!(rpc(addr, &ro, None, call(4, "recall", json!({"query": "x"}))).status, 401);
}

#[test]
fn origin_cors_and_protocol_checks() {
    let opts = ServeOptions { allowed_origins: vec!["https://app.example".into()], ..ServeOptions::default() };
    let (store, addr, _d) = start(opts);
    let (tok, _) = store.create_token("t", Scope::Write, None).unwrap();
    let auth = format!("Bearer {tok}");
    let body = init().to_string();
    let with = |origin: &str| {
        request(
            addr,
            "POST",
            &[("Content-Type", "application/json"), ("Authorization", &auth), ("Origin", origin)],
            &body,
        )
    };
    assert_eq!(with("https://evil.example").status, 403);
    let ok = with("https://app.example");
    assert_eq!(ok.status, 200);
    assert_eq!(ok.header("access-control-allow-origin"), Some("https://app.example"));
    let pre = request(addr, "OPTIONS", &[("Origin", "https://app.example")], "");
    assert_eq!(pre.status, 204);
    assert!(pre.header("access-control-allow-headers").unwrap().contains("Authorization"));
    assert_eq!(request(addr, "OPTIONS", &[("Origin", "https://evil.example")], "").status, 403);
    assert_eq!(request(addr, "GET", &[("Authorization", &auth)], "").status, 405);
    assert_eq!(request(addr, "POST", &[("Authorization", &auth), ("Content-Type", "text/plain")], "{}").status, 415);
    let bad_ver = [
        ("Authorization", auth.as_str()),
        ("Content-Type", "application/json"),
        ("MCP-Protocol-Version", "1999-01-01"),
    ];
    assert_eq!(request(addr, "POST", &bad_ver, &body).status, 400);
    assert_eq!(
        request(addr, "POST", &[("Authorization", &auth), ("Content-Type", "application/json")], "{nope").status,
        400
    );
}

#[test]
fn limits_and_throttling() {
    let opts = ServeOptions { max_body: 1024, ..ServeOptions::default() };
    let (store, addr, _d) = start(opts);
    let (tok, _) = store.create_token("t", Scope::Write, None).unwrap();
    let big = format!("{{\"x\":\"{}\"}}", "a".repeat(4096));
    let auth = format!("Bearer {tok}");
    assert_eq!(
        request(addr, "POST", &[("Authorization", &auth), ("Content-Type", "application/json")], &big).status,
        413
    );
    let fake = format!("syn_{}", "B".repeat(43));
    let mut statuses = Vec::new();
    for _ in 0..25 {
        statuses.push(rpc(addr, &fake, None, init()).status);
    }
    assert_eq!(statuses[0], 401);
    assert_eq!(*statuses.last().unwrap(), 429);
    // Even a valid token is refused from a blocked address.
    assert_eq!(rpc(addr, &tok, None, init()).status, 429);
}

#[test]
fn no_auth_only_on_loopback() {
    let public = ServeOptions { listen: "0.0.0.0:0".into(), no_auth: true, ..ServeOptions::default() };
    assert!(http::bind(&public).is_err());
    let (_store, addr, _d) = start(ServeOptions { no_auth: true, ..ServeOptions::default() });
    let r = request(addr, "POST", &[("Content-Type", "application/json")], &init().to_string());
    assert_eq!(r.status, 200);
}

#[test]
fn healthz_needs_no_token() {
    let (_store, addr, _d) = start(ServeOptions::default());
    let mut s = TcpStream::connect(addr).unwrap();
    s.write_all(b"GET /healthz HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    assert!(out.starts_with("HTTP/1.1 200 OK") && out.ends_with("{\"status\":\"ok\"}"));
}
