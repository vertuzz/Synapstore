//! File downloads through the system `curl`, keeping TLS out of the
//! dependency budget. Secrets are passed on stdin (`--config -`) so they
//! never appear in the process list.

use crate::crypto::{Sha256, hex};
use crate::error::{Error, Result};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

fn http_allowed(url: &str) -> bool {
    ["http://127.0.0.1", "http://localhost", "http://[::1]"].iter().any(|p| url.starts_with(p))
}

/// Download `url` to `dest` with curl. `Ok(None)` on 404. Returns the
/// file's SHA-256, verified against `X-Linked-Etag` when the server sent one.
pub fn fetch(url: &str, dest: &Path, show_progress: bool) -> Result<Option<String>> {
    let headers = dest.with_extension("headers");
    let mut cmd = Command::new("curl");
    cmd.args(["--location", "--silent", "--show-error", "--retry", "3", "--connect-timeout", "30"])
        .args(["--proto-redir", "=https", "--write-out", "%{http_code}", "--config", "-"])
        .arg("--proto")
        .arg(if http_allowed(url) { "=http,https" } else { "=https" })
        .args(["--noproxy", "127.0.0.1,localhost,::1"])
        .arg("--dump-header")
        .arg(&headers)
        .arg("--output")
        .arg(dest)
        .arg(url)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped());
    if show_progress && std::io::IsTerminal::is_terminal(&std::io::stderr()) {
        cmd.arg("--progress-bar").stderr(Stdio::inherit());
    }
    let mut child = cmd.spawn().map_err(|e| {
        Error::InvalidInput(format!("cannot run curl ({e}); download {url} manually and pass its directory"))
    })?;
    {
        // Secrets go through stdin so they never appear in the process list.
        let mut stdin = child.stdin.take().expect("piped stdin");
        if let Ok(tok) = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")) {
            let tok: String = tok.chars().filter(|c| c.is_ascii_graphic()).collect();
            writeln!(stdin, "header = \"Authorization: Bearer {tok}\"")?;
        }
    }
    let out = child.wait_with_output()?;
    let code = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let head = std::fs::read_to_string(&headers).unwrap_or_default();
    std::fs::remove_file(&headers).ok();
    if !out.status.success() {
        return Err(Error::InvalidInput(format!("curl failed for {url}: {}", String::from_utf8_lossy(&out.stderr))));
    }
    match code.as_str() {
        "200" => {}
        "404" => {
            std::fs::remove_file(dest).ok();
            return Ok(None);
        }
        "401" | "403" => {
            return Err(Error::InvalidInput(format!("{url}: HTTP {code} (gated repo? set HF_TOKEN)")));
        }
        _ => return Err(Error::InvalidInput(format!("{url}: HTTP {code}"))),
    }
    let sha = file_sha256(dest)?;
    if let Some(expected) = linked_sha256(&head)
        && expected != sha
    {
        std::fs::remove_file(dest).ok();
        return Err(Error::InvalidInput(format!("{url}: checksum mismatch (expected {expected}, got {sha})")));
    }
    Ok(Some(sha))
}

/// SHA-256 announced by the Hub for LFS files.
pub fn linked_sha256(headers: &str) -> Option<String> {
    headers.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        if !k.trim().eq_ignore_ascii_case("x-linked-etag") {
            return None;
        }
        let v = v.trim().trim_start_matches("W/").trim_matches('"').to_ascii_lowercase();
        (v.len() == 64 && v.chars().all(|c| c.is_ascii_hexdigit())).then_some(v)
    })
}

pub fn file_sha256(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linked_etag() {
        let sha = "a".repeat(64);
        let h = format!("HTTP/2 302\r\nX-Linked-Etag: \"{sha}\"\r\nlocation: x\r\n\r\nHTTP/2 200\r\netag: \"zz\"\r\n");
        assert_eq!(linked_sha256(&h), Some(sha));
        assert_eq!(linked_sha256("etag: \"abc\""), None);
    }
}
