//! Tiny HTTP client for JSON POSTs: plain `http://` in-process, `https://`
//! through the system `curl` (TLS stays out of the dependency budget).
//! Secrets travel on curl's stdin (`--config -`), never in the process list.

use crate::error::{Error, Result};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::time::Duration;

pub fn err(msg: impl Into<String>) -> Error {
    Error::Embedding(msg.into())
}

/// Minimal HTTP/1.1 POST over plain TCP; returns (status, body).
pub fn http_post(url: &str, headers: &[(&str, String)], body: &[u8], timeout: Duration) -> Result<(u16, Vec<u8>)> {
    let rest = url.strip_prefix("http://").ok_or_else(|| err("not an http:// URL"))?;
    let (host, path) = rest.split_once('/').map_or((rest, "/".to_string()), |(h, p)| (h, format!("/{p}")));
    let addr = if host.contains(':') && !host.ends_with(']') { host.to_string() } else { format!("{host}:80") };
    let mut s = TcpStream::connect(&addr).map_err(|e| err(format!("connect {addr}: {e}")))?;
    s.set_read_timeout(Some(timeout))?;
    s.set_write_timeout(Some(timeout))?;
    let mut req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    s.write_all(req.as_bytes())?;
    s.write_all(body)?;
    let mut r = BufReader::new(s);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let status: u16 = line.split_whitespace().nth(1).and_then(|c| c.parse().ok()).ok_or_else(|| err("bad response"))?;
    let (mut len, mut chunked) = (None, false);
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
            if k == "content-length" {
                len = v.parse::<usize>().ok();
            } else if k == "transfer-encoding" && v.eq_ignore_ascii_case("chunked") {
                chunked = true;
            }
        }
    }
    let mut out = Vec::new();
    if chunked {
        loop {
            line.clear();
            r.read_line(&mut line)?;
            let n =
                usize::from_str_radix(line.trim().split(';').next().unwrap_or(""), 16).map_err(|_| err("bad chunk"))?;
            if n == 0 {
                break;
            }
            let start = out.len();
            out.resize(start + n, 0);
            r.read_exact(&mut out[start..])?;
            line.clear();
            r.read_line(&mut line)?;
        }
    } else if let Some(n) = len {
        out.resize(n, 0);
        r.read_exact(&mut out)?;
    } else {
        r.read_to_end(&mut out)?;
    }
    Ok((status, out))
}

/// POST a JSON body with an optional bearer token: plain `http://` in-process,
/// `https://` through curl. Returns (status, body).
pub fn post_json(url: &str, bearer: Option<&str>, body: &str) -> Result<(u16, Vec<u8>)> {
    post_json_timeout(url, bearer, body, Duration::from_secs(120))
}

/// [`post_json`] with an overall time limit.
pub fn post_json_timeout(url: &str, bearer: Option<&str>, body: &str, timeout: Duration) -> Result<(u16, Vec<u8>)> {
    if url.starts_with("http://") {
        let headers: Vec<(&str, String)> =
            bearer.map(|t| ("Authorization", format!("Bearer {t}"))).into_iter().collect();
        http_post(url, &headers, body.as_bytes(), timeout)
    } else if url.starts_with("https://") {
        curl_post(url, bearer, body, timeout)
    } else {
        Err(err(format!("`{url}` is not an http(s) URL")))
    }
}

fn curl_post(url: &str, key: Option<&str>, body: &str, timeout: Duration) -> Result<(u16, Vec<u8>)> {
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let mut cfg = format!("url = \"{}\"\nheader = \"Content-Type: application/json\"\n", esc(url));
    if let Some(k) = key {
        cfg.push_str(&format!("header = \"Authorization: Bearer {}\"\n", esc(k)));
    }
    cfg.push_str(&format!("data-binary = \"{}\"\n", esc(body)));
    let secs = timeout.as_secs().max(1).to_string();
    let mut child = Command::new("curl")
        .args(["--silent", "--show-error", "--proto", "=https", "--connect-timeout", "10", "--max-time", &secs])
        .args(["--write-out", "\n%{http_code}", "--config", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| err(format!("cannot run curl: {e}")))?;
    child.stdin.take().expect("piped").write_all(cfg.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(err(String::from_utf8_lossy(&out.stderr).trim().to_string()));
    }
    let text = out.stdout;
    let split = text.iter().rposition(|&b| b == b'\n').unwrap_or(0);
    let code = String::from_utf8_lossy(&text[split..]).trim().parse().unwrap_or(0);
    Ok((code, text[..split].to_vec()))
}
