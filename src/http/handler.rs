//! Routing for the Streamable HTTP transport: `POST/GET/DELETE /mcp`,
//! CORS pre-flight and `/healthz`. Every `/mcp` request is authenticated
//! with `Authorization: Bearer <token>` against the store.

use super::ServeOptions;
use super::request::{Request, Response};
use super::sessions::{Sessions, Throttle};
use crate::auth::{Scope, TokenInfo};
use crate::mcp::{self, Server, jsonrpc};
use crate::store::SynapStore;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const TOUCH_EVERY: Duration = Duration::from_secs(300);
const SESSION_HEADER: &str = "Mcp-Session-Id";

pub struct App {
    pub store: SynapStore,
    pub opts: ServeOptions,
    pub sessions: Sessions,
    pub throttle: Throttle,
    touched: Mutex<HashMap<[u8; 32], Instant>>,
}

/// Who is calling, after authentication.
struct Caller {
    scope: Scope,
    token: Option<[u8; 32]>,
    info: Option<TokenInfo>,
}

fn rpc_error(status: u16, code: i64, msg: &str) -> Response {
    Response::json(status, &jsonrpc::error(Value::Null, code, msg, None))
}

fn unauthorized(error: &str, desc: &str) -> Response {
    let challenge = format!("Bearer realm=\"synapstore\", error=\"{error}\", error_description=\"{desc}\"");
    Response::json(401, &json!({"error": error, "error_description": desc})).header("WWW-Authenticate", challenge)
}

impl App {
    pub fn new(store: SynapStore, opts: ServeOptions) -> Self {
        App { store, opts, sessions: Sessions::default(), throttle: Throttle::default(), touched: Mutex::default() }
    }

    /// The client address used for throttling.
    pub fn client_ip(&self, req: &Request, peer: IpAddr) -> IpAddr {
        if self.opts.trust_proxy {
            let fwd = req.header("x-forwarded-for").and_then(|v| v.rsplit(',').next()).map(str::trim);
            if let Some(ip) = fwd.or(req.header("x-real-ip")).and_then(|v| v.parse().ok()) {
                return ip;
            }
        }
        peer
    }

    fn origin_allowed(&self, origin: &str) -> bool {
        self.opts.allowed_origins.iter().any(|o| o == "*" || o.eq_ignore_ascii_case(origin))
    }

    fn cors(&self, req: &Request, mut resp: Response) -> Response {
        if let Some(o) = req.header("origin").filter(|o| self.origin_allowed(o)) {
            resp = resp
                .header("Access-Control-Allow-Origin", o)
                .header("Vary", "Origin")
                .header("Access-Control-Expose-Headers", SESSION_HEADER);
        }
        resp
    }

    pub fn handle(&self, req: &Request, ip: IpAddr) -> (Response, Option<String>) {
        let (resp, who) = self.route(req, ip);
        (self.cors(req, resp), who)
    }

    fn route(&self, req: &Request, ip: IpAddr) -> (Response, Option<String>) {
        let path = req.path.trim_end_matches('/');
        match (req.method.as_str(), path) {
            ("GET" | "HEAD", "/healthz") => (Response::json(200, &json!({"status": "ok"})), None),
            ("OPTIONS", "/mcp") => (self.preflight(req), None),
            (_, "/mcp") => {
                // DNS-rebinding defence: browsers always send Origin.
                if let Some(o) = req.header("origin")
                    && !self.origin_allowed(o)
                {
                    return (rpc_error(403, jsonrpc::INVALID_REQUEST, "origin not allowed"), None);
                }
                let caller = match self.authenticate(req, ip) {
                    Ok(c) => c,
                    Err(resp) => return (resp, None),
                };
                let who = caller.info.as_ref().map(|i| i.id.clone());
                let resp = match req.method.as_str() {
                    "POST" => self.post(req, &caller),
                    "DELETE" => match req.header(SESSION_HEADER) {
                        Some(id) if self.sessions.remove(id, caller.token) => Response::new(204),
                        Some(_) => rpc_error(404, jsonrpc::INVALID_REQUEST, "unknown session"),
                        None => rpc_error(400, jsonrpc::INVALID_REQUEST, "missing Mcp-Session-Id"),
                    },
                    // No server-initiated SSE stream is offered (allowed by the spec).
                    _ => Response::new(405).header("Allow", "POST, DELETE"),
                };
                (resp, who)
            }
            _ => (Response::json(404, &json!({"error": "not found"})), None),
        }
    }

    fn preflight(&self, req: &Request) -> Response {
        match req.header("origin") {
            Some(o) if self.origin_allowed(o) => Response::new(204)
                .header("Access-Control-Allow-Methods", "POST, GET, DELETE, OPTIONS")
                .header(
                    "Access-Control-Allow-Headers",
                    "Authorization, Content-Type, Accept, Mcp-Session-Id, Mcp-Protocol-Version, Last-Event-ID",
                )
                .header("Access-Control-Max-Age", "600"),
            _ => Response::new(403),
        }
    }

    fn authenticate(&self, req: &Request, ip: IpAddr) -> Result<Caller, Response> {
        if self.opts.no_auth {
            return Ok(Caller { scope: Scope::Admin, token: None, info: None });
        }
        if let Some(secs) = self.throttle.blocked(ip) {
            return Err(Response::json(429, &json!({"error": "too many failed authentication attempts"}))
                .header("Retry-After", secs.to_string()));
        }
        let header = req.header("authorization").unwrap_or("");
        let Some(secret) = header.strip_prefix("Bearer ").or_else(|| header.strip_prefix("bearer ")) else {
            self.throttle.fail(ip);
            return Err(unauthorized("invalid_request", "missing bearer token"));
        };
        match self.store.authenticate(secret.trim()) {
            Ok(Some((info, hash))) => {
                self.touch(hash);
                Ok(Caller { scope: info.scope, token: Some(hash), info: Some(info) })
            }
            Ok(None) => {
                self.throttle.fail(ip);
                self.store.log().warn("auth_failed", &[("ip", ip.to_string())]);
                Err(unauthorized("invalid_token", "unknown, revoked or expired token"))
            }
            Err(e) => Err(rpc_error(500, jsonrpc::INTERNAL_ERROR, &e.to_string())),
        }
    }

    /// Persist `last_used_at`, at most every few minutes per token.
    fn touch(&self, hash: [u8; 32]) {
        let mut t = self.touched.lock().unwrap_or_else(|p| p.into_inner());
        if t.get(&hash).is_some_and(|at| at.elapsed() < TOUCH_EVERY) {
            return;
        }
        t.insert(hash, Instant::now());
        drop(t);
        if let Err(e) = self.store.touch_token(hash) {
            self.store.log().warn("token_touch", &[("error", e.to_string())]);
        }
    }

    fn post(&self, req: &Request, caller: &Caller) -> Response {
        let ct = req.header("content-type").unwrap_or("").to_ascii_lowercase();
        if !ct.starts_with("application/json") {
            return rpc_error(415, jsonrpc::INVALID_REQUEST, "Content-Type must be application/json");
        }
        if let Some(v) = req.header("mcp-protocol-version")
            && !mcp::PROTOCOL_VERSIONS.contains(&v)
        {
            return rpc_error(400, jsonrpc::INVALID_REQUEST, &format!("unsupported MCP-Protocol-Version {v}"));
        }
        let msg: Value = match serde_json::from_slice(&req.body) {
            Ok(v) => v,
            Err(e) => {
                let body = jsonrpc::error(Value::Null, jsonrpc::PARSE_ERROR, "parse error", Some(json!(e.to_string())));
                return Response::json(400, &body);
            }
        };
        let is_init = |m: &Value| m.get("method").and_then(Value::as_str) == Some("initialize");
        let initializing = is_init(&msg) || msg.as_array().is_some_and(|b| b.iter().any(is_init));
        let (reply, session_id) = if initializing {
            let mut server = Server::scoped(self.store.clone(), caller.scope);
            let reply = server.handle_json(msg);
            (reply, self.sessions.create(server, caller.token))
        } else if let Some(id) = req.header(SESSION_HEADER) {
            let Some(session) = self.sessions.get(id, caller.token) else {
                return rpc_error(404, jsonrpc::INVALID_REQUEST, "unknown or expired session; re-initialize");
            };
            let reply = session.server.lock().unwrap_or_else(|p| p.into_inner()).handle_json(msg);
            (reply, None)
        } else {
            // Stateless use (no session): a fresh server per request.
            (Server::scoped(self.store.clone(), caller.scope).handle_json(msg), None)
        };
        let mut resp = match reply {
            Some(v) => Response::json(200, &v),
            None => Response::new(202),
        };
        if let Some(id) = session_id {
            resp = resp.header(SESSION_HEADER, id);
        }
        resp
    }
}
