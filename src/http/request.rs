//! Minimal, defensive HTTP/1.1 message handling: bounded request lines,
//! header blocks and bodies; `Content-Length` and `chunked` bodies;
//! `Expect: 100-continue`; keep-alive.

use std::io::{self, BufRead, Read, Write};

const MAX_LINE: usize = 8 * 1024;
const MAX_HEADERS: usize = 100;
const MAX_HEADER_BYTES: usize = 32 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    /// The client asked to close the connection after this exchange.
    pub close: bool,
}

impl Request {
    /// Case-insensitive header lookup (first occurrence).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// Why a request could not be read.
#[derive(Debug)]
pub enum ReadError {
    /// Clean EOF before a new request started.
    Closed,
    Io(io::Error),
    /// Malformed or over limits: answer with this status, then close.
    Bad(u16, &'static str),
}

impl From<io::Error> for ReadError {
    fn from(e: io::Error) -> Self {
        ReadError::Io(e)
    }
}

fn read_line(r: &mut impl BufRead, buf: &mut Vec<u8>, budget: &mut usize) -> Result<usize, ReadError> {
    buf.clear();
    let n = r.take((MAX_LINE + 1) as u64).read_until(b'\n', buf)?;
    if n > MAX_LINE || (n > 0 && !buf.ends_with(b"\n")) {
        return Err(ReadError::Bad(431, "header line too long"));
    }
    *budget = budget.checked_sub(n).ok_or(ReadError::Bad(431, "headers too large"))?;
    while buf.last().is_some_and(|b| *b == b'\n' || *b == b'\r') {
        buf.pop();
    }
    Ok(n)
}

/// Read one request. `w` receives `100 Continue` when the client waits.
pub fn read_request(r: &mut impl BufRead, w: &mut impl Write, max_body: usize) -> Result<Request, ReadError> {
    let mut line = Vec::new();
    let mut budget = MAX_HEADER_BYTES;
    // Tolerate empty lines between pipelined requests (RFC 9112 §2.2).
    loop {
        if read_line(r, &mut line, &mut budget)? == 0 {
            return Err(ReadError::Closed);
        }
        if !line.is_empty() {
            break;
        }
    }
    let text = String::from_utf8(line.clone()).map_err(|_| ReadError::Bad(400, "request line is not UTF-8"))?;
    let mut parts = text.split(' ');
    let (Some(method), Some(target), Some(version), None) = (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(ReadError::Bad(400, "malformed request line"));
    };
    if !method.bytes().all(|b| b.is_ascii_uppercase()) || method.is_empty() {
        return Err(ReadError::Bad(400, "malformed method"));
    }
    let http10 = match version {
        "HTTP/1.1" => false,
        "HTTP/1.0" => true,
        _ => return Err(ReadError::Bad(505, "only HTTP/1.x is supported")),
    };
    let (method, path, http10) = (method.to_string(), target.split('?').next().unwrap_or("").to_string(), http10);
    let mut headers = Vec::new();
    loop {
        if read_line(r, &mut line, &mut budget)? == 0 {
            return Err(ReadError::Bad(400, "unexpected end of headers"));
        }
        if line.is_empty() {
            break;
        }
        if headers.len() >= MAX_HEADERS {
            return Err(ReadError::Bad(431, "too many headers"));
        }
        let h = std::str::from_utf8(&line).map_err(|_| ReadError::Bad(400, "header is not UTF-8"))?;
        let (k, v) = h.split_once(':').ok_or(ReadError::Bad(400, "malformed header"))?;
        if k.is_empty() || k.bytes().any(|b| b.is_ascii_whitespace()) {
            return Err(ReadError::Bad(400, "malformed header name"));
        }
        headers.push((k.to_string(), v.trim().to_string()));
    }
    let mut req = Request { method, path, headers, body: Vec::new(), close: http10 };
    if let Some(c) = req.header("connection") {
        let c = c.to_ascii_lowercase();
        req.close = c.contains("close") || (http10 && !c.contains("keep-alive"));
    }
    let chunked = req.header("transfer-encoding").map(|t| t.to_ascii_lowercase());
    let length = req.header("content-length").map(str::to_string);
    if chunked.is_some() && length.is_some() {
        return Err(ReadError::Bad(400, "both Content-Length and Transfer-Encoding"));
    }
    if req.header("expect").is_some_and(|e| e.eq_ignore_ascii_case("100-continue")) {
        w.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
        w.flush()?;
    }
    match (chunked, length) {
        (Some(t), _) if t == "chunked" => req.body = read_chunked(r, max_body)?,
        (Some(_), _) => return Err(ReadError::Bad(501, "unsupported transfer coding")),
        (None, Some(l)) => {
            let n: usize = l.parse().map_err(|_| ReadError::Bad(400, "bad Content-Length"))?;
            if n > max_body {
                return Err(ReadError::Bad(413, "request body too large"));
            }
            req.body = vec![0; n];
            r.read_exact(&mut req.body)?;
        }
        (None, None) => {}
    }
    Ok(req)
}

fn read_chunked(r: &mut impl BufRead, max_body: usize) -> Result<Vec<u8>, ReadError> {
    let mut body = Vec::new();
    let mut line = Vec::new();
    let mut budget = MAX_HEADER_BYTES * 4;
    loop {
        read_line(r, &mut line, &mut budget)?;
        let size = std::str::from_utf8(&line).ok().and_then(|l| l.split(';').next()).map(str::trim);
        let n = size.and_then(|s| usize::from_str_radix(s, 16).ok()).ok_or(ReadError::Bad(400, "bad chunk size"))?;
        if n == 0 {
            // Skip trailers.
            while read_line(r, &mut line, &mut budget)? > 0 && !line.is_empty() {}
            return Ok(body);
        }
        if body.len() + n > max_body {
            return Err(ReadError::Bad(413, "request body too large"));
        }
        let start = body.len();
        body.resize(start + n, 0);
        r.read_exact(&mut body[start..])?;
        read_line(r, &mut line, &mut budget)?;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn new(status: u16) -> Self {
        Response { status, headers: Vec::new(), body: Vec::new() }
    }
    pub fn json(status: u16, v: &serde_json::Value) -> Self {
        Response::new(status).header("Content-Type", "application/json").body(v.to_string().into_bytes())
    }
    pub fn header(mut self, k: &str, v: impl Into<String>) -> Self {
        self.headers.push((k.into(), v.into()));
        self
    }
    pub fn body(mut self, b: Vec<u8>) -> Self {
        self.body = b;
        self
    }
}

pub fn reason(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        200 => "OK",
        202 => "Accepted",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        406 => "Not Acceptable",
        408 => "Request Timeout",
        413 => "Content Too Large",
        415 => "Unsupported Media Type",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        _ => "Unknown",
    }
}

/// Serialize `resp`; security headers are added to every response.
pub fn write_response(w: &mut impl Write, resp: &Response, close: bool) -> io::Result<()> {
    let mut head = format!("HTTP/1.1 {} {}\r\n", resp.status, reason(resp.status));
    for (k, v) in &resp.headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n");
    head.push_str(&format!("Content-Length: {}\r\n", resp.body.len()));
    if close {
        head.push_str("Connection: close\r\n");
    }
    head.push_str("\r\n");
    w.write_all(head.as_bytes())?;
    w.write_all(&resp.body)?;
    w.flush()
}

#[cfg(test)]
mod tests;
