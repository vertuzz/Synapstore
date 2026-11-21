use super::*;

fn parse(raw: &str) -> Result<Request, ReadError> {
    read_request(&mut raw.as_bytes(), &mut Vec::new(), 1024)
}

#[test]
fn parses_requests() {
    let r =
        parse("POST /mcp?x=1 HTTP/1.1\r\nHost: a\r\nContent-Length: 2\r\nAUTHORIZATION: Bearer t\r\n\r\n{}").unwrap();
    assert_eq!((r.method.as_str(), r.path.as_str(), r.body.as_slice()), ("POST", "/mcp", b"{}".as_slice()));
    assert_eq!(r.header("authorization"), Some("Bearer t"));
    assert!(!r.close);
    let r = parse("GET / HTTP/1.0\r\n\r\n").unwrap();
    assert!(r.close);
    let r = parse(
        "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n3\r\nabc\r\n2;x\r\nde\r\n0\r\n\r\n",
    );
    let r = r.unwrap();
    assert_eq!(r.body, b"abcde");
    assert!(r.close);
}

#[test]
fn sends_100_continue() {
    let mut out = Vec::new();
    let raw = "POST / HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 1\r\n\r\nx";
    read_request(&mut raw.as_bytes(), &mut out, 10).unwrap();
    assert_eq!(out, b"HTTP/1.1 100 Continue\r\n\r\n");
}

#[test]
fn rejects_bad_input() {
    let bad = |raw: &str| match parse(raw) {
        Err(ReadError::Bad(code, _)) => code,
        other => panic!("{raw:?} -> {other:?}"),
    };
    assert_eq!(bad("POST / HTTP/1.1\r\nContent-Length: 5000\r\n\r\n"), 413);
    assert_eq!(bad("GET / HTTP/2\r\n\r\n"), 505);
    assert_eq!(bad("GET /\r\n\r\n"), 400);
    assert_eq!(bad("GET / HTTP/1.1\r\nno-colon\r\n\r\n"), 400);
    assert_eq!(bad("POST / HTTP/1.1\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\n"), 400);
    assert_eq!(bad(&format!("GET /{} HTTP/1.1\r\n\r\n", "a".repeat(9000))), 431);
    let many: String = (0..101).map(|i| format!("h{i}: v\r\n")).collect();
    assert_eq!(bad(&format!("GET / HTTP/1.1\r\n{many}\r\n")), 431);
    assert!(matches!(parse(""), Err(ReadError::Closed)));
}

#[test]
fn writes_responses() {
    let mut out = Vec::new();
    write_response(&mut out, &Response::new(204).header("A", "b"), true).unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.starts_with("HTTP/1.1 204 No Content\r\nA: b\r\n"));
    assert!(s.contains("Content-Length: 0\r\n") && s.contains("Connection: close\r\n") && s.ends_with("\r\n\r\n"));
}
