//! Memory commands for agents and people (`remember`, `recall`, `graph`…).
//! Each one is an MCP tool call, run on the local store or, with `--url`,
//! against a remote `synapstore serve` (token from `SYNAPSTORE_TOKEN`).

use super::view::{self, View};
use crate::SynapStore;
use crate::mcp::handlers::{self, Call};
use serde_json::{Map, Value, json};

/// Flags shared by the memory commands.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct AgentFlags {
    pub facts: Vec<String>,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub kind: Option<String>,
    pub top_k: Option<u64>,
    pub limit: Option<u64>,
    pub depth: Option<u64>,
    pub max_age: Option<String>,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub json: bool,
    pub url: Option<String>,
}

/// One tool call plus how to print its result.
#[derive(Debug, Clone, PartialEq)]
pub struct Op {
    pub tool: String,
    pub args: Value,
    pub view: View,
    pub json: bool,
    pub url: Option<String>,
    /// Read `args.text` from stdin (the text was `-`).
    pub stdin_text: bool,
}

pub const VERBS: &[&str] =
    &["remember", "recall", "log", "events", "graph", "facts", "fact", "forget", "explain", "call"];

fn parse_fact(s: &str) -> Result<Value, String> {
    let parts: Vec<&str> = s.split('|').map(str::trim).collect();
    match parts.as_slice() {
        [s, p, o] if !s.is_empty() && !p.is_empty() && !o.is_empty() => Ok(json!([s, p, o])),
        _ => Err(format!("--fact `{s}`: expected \"subject|predicate|object\"")),
    }
}

fn put(a: &mut Map<String, Value>, k: &str, v: Option<Value>) {
    if let Some(v) = v {
        a.insert(k.into(), v);
    }
}

fn need(pos: &mut std::vec::IntoIter<String>, what: &str) -> Result<String, String> {
    pos.next().ok_or_else(|| format!("missing {what}"))
}

/// Build the tool call for `verb` (pure).
pub fn build(verb: &str, positional: Vec<String>, f: AgentFlags) -> Result<Op, String> {
    let mut pos = positional.into_iter();
    let mut a = Map::new();
    let tags = (!f.tags.is_empty()).then(|| json!(f.tags));
    let call_name: String;
    let (tool, view, mut stdin_text) = match verb {
        "remember" => {
            let text = need(&mut pos, "memory text (or `-` to read stdin)")?;
            let facts = f.facts.iter().map(|s| parse_fact(s)).collect::<Result<Vec<_>, _>>()?;
            let stdin = text == "-";
            put(&mut a, "text", Some(json!(text)));
            put(&mut a, "facts", (!facts.is_empty()).then(|| json!(facts)));
            put(&mut a, "tags", tags);
            put(&mut a, "source", f.source.clone().map(Value::from));
            put(&mut a, "kind", f.kind.clone().map(Value::from));
            ("memorize", View::Remember, stdin)
        }
        "recall" => {
            let q: Vec<String> = pos.by_ref().collect();
            if q.is_empty() {
                return Err("recall needs a query".into());
            }
            put(&mut a, "query", Some(json!(q.join(" "))));
            put(&mut a, "top_k", f.top_k.or(f.limit).map(Value::from));
            put(&mut a, "max_age", f.max_age.clone().map(Value::from));
            put(&mut a, "tags", tags);
            ("recall", View::Recall, false)
        }
        "log" => {
            let text = need(&mut pos, "event text (or `-`)")?;
            let stdin = text == "-";
            put(&mut a, "text", Some(json!(text)));
            put(&mut a, "kind", f.kind.clone().map(Value::from));
            ("log_event", View::Logged, stdin)
        }
        "events" => {
            put(&mut a, "limit", f.limit.or(f.top_k).map(Value::from));
            put(&mut a, "kind", f.kind.clone().map(Value::from));
            ("recent_events", View::Events, false)
        }
        "graph" => {
            let q: Vec<String> = pos.by_ref().collect();
            if q.is_empty() {
                return Err("graph needs an entity name".into());
            }
            put(&mut a, "subject", Some(json!(q.join(" "))));
            put(&mut a, "depth", f.depth.map(Value::from));
            put(&mut a, "limit", f.limit.map(Value::from));
            ("graph_query", View::Graph, false)
        }
        "facts" => {
            put(&mut a, "subject", f.subject.clone().map(Value::from));
            put(&mut a, "predicate", f.predicate.clone().map(Value::from));
            put(&mut a, "object", f.object.clone().map(Value::from));
            put(&mut a, "limit", f.limit.map(Value::from));
            ("graph_match", View::Facts, false)
        }
        "fact" => {
            let sub = need(&mut pos, "`add` or `rm`")?;
            let (s, p, o) = (need(&mut pos, "subject")?, need(&mut pos, "predicate")?, need(&mut pos, "object")?);
            match sub.as_str() {
                "add" => {
                    put(&mut a, "facts", Some(json!([[s, p, o]])));
                    ("add_facts", View::Json, false)
                }
                "rm" | "remove" | "retract" => {
                    put(&mut a, "subject", Some(json!(s)));
                    put(&mut a, "predicate", Some(json!(p)));
                    put(&mut a, "object", Some(json!(o)));
                    ("retract_fact", View::Json, false)
                }
                other => return Err(format!("unknown fact subcommand `{other}` (add, rm)")),
            }
        }
        "forget" => {
            let ids: Vec<String> = pos.by_ref().collect();
            if ids.is_empty() {
                return Err("forget needs one or more memory ids".into());
            }
            put(&mut a, "ids", Some(json!(ids)));
            ("forget", View::Forgot, false)
        }
        "explain" => {
            let id = need(&mut pos, "memory id")?;
            let q: Vec<String> = pos.by_ref().collect();
            put(&mut a, "target_id", Some(json!(id)));
            put(&mut a, "query", (!q.is_empty()).then(|| json!(q.join(" "))));
            ("explain", View::Json, false)
        }
        "call" => {
            let name = need(&mut pos, "tool name")?;
            if let Some(raw) = pos.next() {
                let v: Value = serde_json::from_str(&raw).map_err(|e| format!("call: arguments are not JSON: {e}"))?;
                let Value::Object(m) = v else { return Err("call: arguments must be a JSON object".into()) };
                a.extend(m);
            }
            call_name = name;
            (call_name.as_str(), View::Json, false)
        }
        other => return Err(format!("unknown command `{other}`")),
    };
    if let Some(extra) = pos.next() {
        return Err(format!("unexpected argument `{extra}` (quote multi-word text)"));
    }
    stdin_text &= verb != "call";
    Ok(Op { tool: tool.into(), args: Value::Object(a), view, json: f.json, url: f.url, stdin_text })
}

/// `https://host` → `https://host/mcp`; explicit paths are kept.
pub fn endpoint(url: &str) -> String {
    let u = url.trim_end_matches('/');
    let after_scheme = u.split_once("://").map_or(u, |(_, r)| r);
    if after_scheme.contains('/') { u.to_string() } else { format!("{u}/mcp") }
}

fn remote_call(url: &str, token: Option<String>, tool: &str, args: &Value) -> Result<Value, String> {
    let token = std::env::var("SYNAPSTORE_TOKEN").ok().filter(|t| !t.is_empty()).or(token);
    let body = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": args}});
    let url = endpoint(url);
    let (status, raw) = crate::embed::remote::post_json(&url, token.as_deref(), &body.to_string())
        .map_err(|e| e.to_string().replace("invalid input: embedding endpoint: ", ""))?;
    match status {
        200 => {}
        401 if token.is_none() => return Err(format!("{url}: unauthorized; set SYNAPSTORE_TOKEN")),
        401 => return Err(format!("{url}: token rejected (revoked or expired?)")),
        429 => return Err(format!("{url}: too many failed attempts; wait a few minutes")),
        s => return Err(format!("{url}: HTTP {s}: {}", String::from_utf8_lossy(&raw).trim())),
    }
    let v: Value = serde_json::from_slice(&raw).map_err(|e| format!("{url}: bad response: {e}"))?;
    if let Some(e) = v.get("error") {
        return Err(e["message"].as_str().unwrap_or("error").to_string());
    }
    let r = &v["result"];
    if r["isError"] == true {
        return Err(r["content"][0]["text"].as_str().unwrap_or("tool error").to_string());
    }
    Ok(r["structuredContent"].clone())
}

/// Run the tool call; `open` opens the local store when no URL is set.
/// `token` is used for a remote server when `SYNAPSTORE_TOKEN` is unset.
pub fn execute(
    mut op: Op,
    token: Option<String>,
    open: impl FnOnce() -> Result<SynapStore, String>,
) -> Result<(), String> {
    if op.stdin_text {
        let mut text = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut text).map_err(|e| e.to_string())?;
        if text.trim().is_empty() {
            return Err("stdin was empty".into());
        }
        op.args["text"] = json!(text.trim_end());
    }
    let url = op.url.clone().or_else(|| std::env::var("SYNAPSTORE_URL").ok().filter(|u| !u.is_empty()));
    let result = match url {
        Some(u) => remote_call(&u, token, &op.tool, &op.args)?,
        None => match handlers::call(&open()?, &op.tool, &op.args) {
            Call::Done(Ok(v)) => v,
            Call::Done(Err(e)) => return Err(format!("{} error: {e}", e.kind())),
            Call::UnknownTool => return Err(format!("unknown tool `{}`", op.tool)),
        },
    };
    let text = if op.json {
        serde_json::to_string_pretty(&result).unwrap_or_default()
    } else {
        view::render(op.view, &result)
    };
    super::run::print(&text);
    Ok(())
}
