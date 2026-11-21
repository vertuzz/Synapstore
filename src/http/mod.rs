//! MCP over HTTP ("Streamable HTTP" transport, MCP 2025-03-26+) on plain
//! `std::net`, for exposing one memory file to remote agents.
//!
//! * Bearer-token authentication (see [`crate::auth`]) on every request,
//!   with per-address throttling of failed attempts.
//! * `Origin` validation against an allow-list (DNS-rebinding defence).
//! * Bounded request sizes, header counts, idle timeouts and worker pool.
//!
//! TLS is not built in (it would break the dependency budget): put a
//! TLS-terminating reverse proxy such as Caddy or nginx in front of it.

pub mod handler;
pub mod request;
pub mod sessions;

use crate::store::SynapStore;
use handler::App;
use request::{ReadError, Response, read_request, write_response};
use std::io::{BufReader, BufWriter, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

const IDLE_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_REQUESTS_PER_CONNECTION: usize = 10_000;

#[derive(Debug, Clone, PartialEq)]
pub struct ServeOptions {
    pub listen: String,
    /// Browser origins allowed to call the server (`*` for any).
    pub allowed_origins: Vec<String>,
    pub threads: usize,
    pub max_body: usize,
    /// Disable authentication (only allowed on loopback addresses).
    pub no_auth: bool,
    /// Take the client address from `X-Forwarded-For` (behind a proxy).
    pub trust_proxy: bool,
}

impl Default for ServeOptions {
    fn default() -> Self {
        ServeOptions {
            listen: "127.0.0.1:8765".into(),
            allowed_origins: Vec::new(),
            threads: 16,
            max_body: 8 << 20,
            no_auth: false,
            trust_proxy: false,
        }
    }
}

/// Bind the listening socket, enforcing the "no auth only on loopback" rule.
pub fn bind(opts: &ServeOptions) -> std::io::Result<TcpListener> {
    let addrs: Vec<SocketAddr> = opts.listen.to_socket_addrs()?.collect();
    if opts.no_auth && addrs.iter().any(|a| !a.ip().is_loopback()) {
        return Err(std::io::Error::other("--no-auth is only allowed on a loopback address (127.0.0.1 / ::1)"));
    }
    TcpListener::bind(&addrs[..])
}

/// Serve until the process is stopped.
pub fn serve(store: SynapStore, opts: ServeOptions, listener: TcpListener) -> std::io::Result<()> {
    let app = Arc::new(App::new(store, opts));
    let threads = app.opts.threads.max(1);
    let (tx, rx) = mpsc::sync_channel::<TcpStream>(threads * 4);
    let rx = Arc::new(Mutex::new(rx));
    for i in 0..threads {
        let (app, rx) = (app.clone(), rx.clone());
        std::thread::Builder::new().name(format!("synapstore-http-{i}")).spawn(move || {
            loop {
                let next = rx.lock().unwrap_or_else(|p| p.into_inner()).recv();
                match next {
                    Ok(stream) => connection(&app, stream),
                    Err(_) => break,
                }
            }
        })?;
    }
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        if let Err(mpsc::TrySendError::Full(mut s)) = tx.try_send(stream) {
            s.set_write_timeout(Some(Duration::from_secs(2))).ok();
            let busy = Response::json(503, &serde_json::json!({"error": "server busy"})).header("Retry-After", "1");
            write_response(&mut s, &busy, true).ok();
        }
    }
    Ok(())
}

fn connection(app: &App, stream: TcpStream) {
    let peer = stream.peer_addr().map(|a| a.ip()).unwrap_or(IpAddr::from([0, 0, 0, 0]));
    stream.set_read_timeout(Some(IDLE_TIMEOUT)).ok();
    stream.set_write_timeout(Some(IDLE_TIMEOUT)).ok();
    stream.set_nodelay(true).ok();
    let Ok(write_half) = stream.try_clone() else { return };
    let mut reader = BufReader::new(stream);
    let mut writer = BufWriter::new(write_half);
    for _ in 0..MAX_REQUESTS_PER_CONNECTION {
        let start = Instant::now();
        let req = match read_request(&mut reader, &mut writer, app.opts.max_body) {
            Ok(r) => r,
            Err(ReadError::Bad(status, msg)) => {
                let body = serde_json::json!({"error": msg});
                write_response(&mut writer, &Response::json(status, &body), true).ok();
                return;
            }
            Err(ReadError::Closed | ReadError::Io(_)) => return,
        };
        let ip = app.client_ip(&req, peer);
        let (resp, who) = app.handle(&req, ip);
        let mut fields = vec![
            ("method", req.method.clone()),
            ("path", req.path.clone()),
            ("status", resp.status.to_string()),
            ("ip", ip.to_string()),
            ("latency_us", start.elapsed().as_micros().to_string()),
        ];
        if let Some(id) = who {
            fields.push(("token", id));
        }
        app.store.log().debug("http", &fields);
        let head_only = req.method == "HEAD";
        let resp = if head_only { Response { body: Vec::new(), ..resp } } else { resp };
        if write_response(&mut writer, &resp, req.close).is_err() || req.close {
            return;
        }
        writer.flush().ok();
    }
}
