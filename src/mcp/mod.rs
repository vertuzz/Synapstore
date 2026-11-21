//! Model Context Protocol server over stdio (PRD §5): newline-delimited
//! JSON-RPC 2.0 on stdin/stdout. stdout carries protocol messages only.

pub mod args;
pub mod handlers;
pub mod jsonrpc;
pub mod tools;

use crate::auth::{Scope, tool_scope};
use crate::store::SynapStore;
use handlers::Call;
use jsonrpc::{Incoming, classify, error, success};
use serde_json::{Value, json};
use std::io::{BufRead, Write};

/// Protocol revisions this server speaks, newest first.
pub const PROTOCOL_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

const INSTRUCTIONS: &str = "SynapStore is your long-term memory. Use `memorize` to store facts, decisions and \
observations, `recall` to search semantically (fresh memories rank higher), `graph_query` to get everything known \
about an entity, and `log_event` / `recent_events` for the episodic timeline. Use `explain` when an expected memory \
is not recalled.\n\nThe knowledge graph is only as good as the `facts` you pass: the store never extracts them from \
text. Whenever a memory states a durable relation (people and their roles, family, places, employers, ownership, \
projects and their stack, key dates), add it to `memorize` as [subject, predicate, object]. Reuse existing entity \
names (`graph_match` shows them) and short snake_case predicates (works_at, parent_of, lives_in, owns, uses). Do not \
turn fast-changing state (prices, daily status) into facts. When something changes or was wrong, memorize the \
corrected text with the old memory's id in `supersedes`: the old memory and its facts are removed atomically. For a \
fact added without a memory, `retract_fact` the old one before adding the new one.";

pub struct Server {
    store: SynapStore,
    protocol: Option<String>,
    /// Permissions of the caller; `None` = unrestricted (stdio).
    scope: Option<Scope>,
}

/// Pick the protocol version to answer `initialize` with.
pub fn negotiate(requested: Option<&str>) -> &'static str {
    requested.and_then(|r| PROTOCOL_VERSIONS.iter().find(|v| **v == r).copied()).unwrap_or(PROTOCOL_VERSIONS[0])
}

impl Server {
    pub fn new(store: SynapStore) -> Self {
        Server { store, protocol: None, scope: None }
    }

    /// A server whose tools are limited to what `scope` allows.
    pub fn scoped(store: SynapStore, scope: Scope) -> Self {
        Server { store, protocol: None, scope: Some(scope) }
    }

    fn allowed(&self, tool: &str) -> bool {
        self.scope.is_none_or(|s| s >= tool_scope(tool))
    }

    pub fn protocol(&self) -> Option<&str> {
        self.protocol.as_deref()
    }

    /// Handle one raw input line; returns the response line, if any.
    pub fn handle_line(&mut self, line: &str) -> Option<String> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }
        let out = match serde_json::from_str::<Value>(line) {
            Err(e) => Some(error(Value::Null, jsonrpc::PARSE_ERROR, "parse error", Some(json!(e.to_string())))),
            Ok(v) => self.handle_json(v),
        };
        out.map(|v| v.to_string())
    }

    /// Handle one decoded message or batch; `None` if nothing to reply.
    pub fn handle_json(&mut self, v: Value) -> Option<Value> {
        match v {
            Value::Array(batch) if batch.is_empty() => {
                Some(error(Value::Null, jsonrpc::INVALID_REQUEST, "empty batch", None))
            }
            Value::Array(batch) => {
                let replies: Vec<Value> = batch.into_iter().filter_map(|m| self.handle_value(m)).collect();
                (!replies.is_empty()).then_some(Value::Array(replies))
            }
            v => self.handle_value(v),
        }
    }

    /// Handle one decoded message.
    pub fn handle_value(&mut self, v: Value) -> Option<Value> {
        match classify(v) {
            Incoming::Invalid(resp) => Some(resp),
            Incoming::Response => None,
            Incoming::Notification { method, .. } => {
                self.store.log().debug("notification", &[("method", method)]);
                None
            }
            Incoming::Request { id, method, params } => Some(self.request(id, &method, &params)),
        }
    }

    fn request(&mut self, id: Value, method: &str, params: &Value) -> Value {
        match method {
            "initialize" => {
                let v = negotiate(params.get("protocolVersion").and_then(Value::as_str));
                self.protocol = Some(v.to_string());
                success(
                    id,
                    json!({
                        "protocolVersion": v,
                        "capabilities": {"tools": {"listChanged": false}},
                        "serverInfo": {"name": "synapstore", "title": "SynapStore", "version": env!("CARGO_PKG_VERSION")},
                        "instructions": INSTRUCTIONS,
                    }),
                )
            }
            "ping" => success(id, json!({})),
            "tools/list" => {
                let all = tools::definitions();
                let visible: Vec<Value> = all
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|t| t["name"].as_str().is_some_and(|n| self.allowed(n)))
                    .cloned()
                    .collect();
                success(id, json!({"tools": visible}))
            }
            "resources/list" => success(id, json!({"resources": []})),
            "resources/templates/list" => success(id, json!({"resourceTemplates": []})),
            "prompts/list" => success(id, json!({"prompts": []})),
            "logging/setLevel" => success(id, json!({})),
            "tools/call" => self.tools_call(id, params),
            _ => error(id, jsonrpc::METHOD_NOT_FOUND, &format!("method not found: {method}"), None),
        }
    }

    fn tools_call(&mut self, id: Value, params: &Value) -> Value {
        let Some(name) = params.get("name").and_then(Value::as_str) else {
            return error(id, jsonrpc::INVALID_PARAMS, "tools/call requires `name`", None);
        };
        let args = params.get("arguments").cloned().unwrap_or(Value::Null);
        if tools::definitions().as_array().is_some_and(|d| d.iter().any(|t| t["name"] == name)) && !self.allowed(name) {
            let need = tool_scope(name).as_str();
            let msg = format!("tool `{name}` needs a token with `{need}` scope");
            return success(
                id,
                json!({
                    "content": [{"type": "text", "text": format!("forbidden error: {msg}")}],
                    "structuredContent": {"error": {"kind": "forbidden", "message": msg}},
                    "isError": true,
                }),
            );
        }
        match handlers::call(&self.store, name, &args) {
            Call::UnknownTool => error(id, jsonrpc::INVALID_PARAMS, &format!("unknown tool: {name}"), None),
            Call::Done(Ok(value)) => {
                let text = serde_json::to_string_pretty(&value).unwrap_or_default();
                success(
                    id,
                    json!({
                        "content": [{"type": "text", "text": text}],
                        "structuredContent": value,
                        "isError": false,
                    }),
                )
            }
            Call::Done(Err(e)) => success(
                id,
                json!({
                    "content": [{"type": "text", "text": format!("{} error: {e}", e.kind())}],
                    "structuredContent": {"error": {"kind": e.kind(), "message": e.to_string()}},
                    "isError": true,
                }),
            ),
        }
    }
}

/// Serve MCP until `input` reaches EOF.
/// Invalid UTF-8 is answered with a parse error rather than ending the session.
pub fn serve(store: SynapStore, mut input: impl BufRead, mut output: impl Write) -> std::io::Result<()> {
    let mut server = Server::new(store);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        if input.read_until(b'\n', &mut buf)? == 0 {
            break;
        }
        let line = String::from_utf8_lossy(&buf);
        if let Some(reply) = server.handle_line(&line) {
            output.write_all(reply.as_bytes())?;
            output.write_all(b"\n")?;
            output.flush()?;
        }
    }
    Ok(())
}
