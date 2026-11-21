//! Minimal edits to JSON / JSONC config files that keep the user's
//! formatting, key order and comments: find a member's byte span and splice.

use serde_json::Value;

struct Member {
    key: String,
    key_start: usize,
    value: (usize, usize),
    /// Index just past the member's trailing comma, if it has one.
    comma_end: Option<usize>,
}

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    loop {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if b[i..].starts_with(b"//") {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i..].starts_with(b"/*") {
            i = b[i + 2..].windows(2).position(|w| w == b"*/").map_or(b.len(), |p| i + 2 + p + 2);
        } else {
            return i;
        }
    }
}

fn skip_string(b: &[u8], mut i: usize) -> Result<usize, String> {
    i += 1;
    while i < b.len() {
        match b[i] {
            b'\\' => i += 2,
            b'"' => return Ok(i + 1),
            _ => i += 1,
        }
    }
    Err("unterminated string".into())
}

/// End (exclusive) of the value starting at `i`.
fn skip_value(b: &[u8], i: usize) -> Result<usize, String> {
    match b.get(i) {
        Some(b'"') => skip_string(b, i),
        Some(b'{' | b'[') => {
            let (mut depth, mut j) = (0usize, i);
            while j < b.len() {
                match b[j] {
                    b'"' => {
                        j = skip_string(b, j)?;
                        continue;
                    }
                    b'/' if matches!(b.get(j + 1), Some(b'/' | b'*')) => {
                        j = skip_ws(b, j);
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Ok(j + 1);
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            Err("unbalanced brackets".into())
        }
        Some(_) => {
            let mut j = i;
            while j < b.len() && !matches!(b[j], b',' | b'}' | b']' | b'/') && !b[j].is_ascii_whitespace() {
                j += 1;
            }
            if j == i { Err(format!("expected a value at byte {i}")) } else { Ok(j) }
        }
        None => Err("unexpected end of file".into()),
    }
}

/// Members of the object at `open` (a `{`) and the index of its `}`.
fn members(b: &[u8], open: usize) -> Result<(Vec<Member>, usize), String> {
    let mut out = Vec::new();
    let mut i = open + 1;
    loop {
        i = skip_ws(b, i);
        match b.get(i) {
            Some(b'}') => return Ok((out, i)),
            Some(b'"') => {}
            _ => return Err(format!("expected a key at byte {i}")),
        }
        let key_end = skip_string(b, i)?;
        let key: String = serde_json::from_slice(&b[i..key_end]).map_err(|e| e.to_string())?;
        let colon = skip_ws(b, key_end);
        if b.get(colon) != Some(&b':') {
            return Err(format!("expected `:` at byte {colon}"));
        }
        let vs = skip_ws(b, colon + 1);
        let ve = skip_value(b, vs)?;
        let after = skip_ws(b, ve);
        let comma_end = (b.get(after) == Some(&b',')).then_some(after + 1);
        out.push(Member { key, key_start: i, value: (vs, ve), comma_end });
        i = comma_end.unwrap_or(after);
    }
}

fn line_indent(t: &str, at: usize) -> String {
    let start = t[..at].rfind('\n').map_or(0, |p| p + 1);
    t[start..at].chars().take_while(|c| *c == ' ' || *c == '\t').collect()
}

fn pretty(v: &Value, indent: &str) -> String {
    serde_json::to_string_pretty(v).unwrap_or_default().replace('\n', &format!("\n{indent}"))
}

fn root(t: &str) -> Result<usize, String> {
    let i = skip_ws(t.as_bytes(), 0);
    if t.as_bytes().get(i) == Some(&b'{') { Ok(i) } else { Err("the file is not a JSON object".into()) }
}

/// Set `path.../key` to `value`, creating parent objects as needed.
pub fn upsert(text: &str, path: &[&str], key: &str, value: &Value) -> Result<String, String> {
    let mut t = if text.trim().is_empty() { "{}\n".to_string() } else { text.to_string() };
    let strict = serde_json::from_str::<Value>(&t).is_ok();
    let mut obj = root(&t)?;
    let full: Vec<&str> = path.iter().copied().chain([key]).collect();
    for (depth, name) in full.iter().enumerate() {
        let (ms, close) = members(t.as_bytes(), obj)?;
        let last = depth + 1 == full.len();
        if let Some(m) = ms.iter().find(|m| m.key == *name) {
            if last {
                let indent = line_indent(&t, m.key_start);
                t.replace_range(m.value.0..m.value.1, &pretty(value, &indent));
                break;
            }
            if t.as_bytes()[m.value.0] != b'{' {
                return Err(format!("`{name}` is not an object"));
            }
            obj = m.value.0;
            continue;
        }
        // Missing: insert `name` holding the rest of the path.
        let mut v = value.clone();
        for k in full[depth + 1..].iter().rev() {
            v = serde_json::json!({ *k: v });
        }
        let close_indent = line_indent(&t, close);
        // Siblings' indentation, unless the object is written on one line.
        let indent = match ms.first() {
            Some(m) if t[obj..m.key_start].contains('\n') => line_indent(&t, m.key_start),
            _ => format!("{close_indent}  "),
        };
        let member = format!("{}: {}", serde_json::to_string(name).unwrap_or_default(), pretty(&v, &indent));
        match ms.last() {
            Some(m) if m.comma_end.is_some() => t.insert_str(m.comma_end.unwrap_or(0), &format!("\n{indent}{member},")),
            Some(m) => t.insert_str(m.value.1, &format!(",\n{indent}{member}")),
            None if t[obj + 1..close].trim().is_empty() => {
                t.replace_range(obj + 1..close, &format!("\n{indent}{member}\n{close_indent}"));
            }
            None => t.insert_str(obj + 1, &format!("\n{indent}{member}")),
        }
        break;
    }
    if strict && serde_json::from_str::<Value>(&t).is_err() {
        return Err("internal error: the edit would corrupt the file".into());
    }
    Ok(t)
}

/// Remove `path.../key`; `None` when it is not there.
pub fn remove(text: &str, path: &[&str], key: &str) -> Result<Option<String>, String> {
    if text.trim().is_empty() {
        return Ok(None);
    }
    let mut t = text.to_string();
    let mut obj = root(&t)?;
    for name in path {
        let (ms, _) = members(t.as_bytes(), obj)?;
        match ms.iter().find(|m| m.key == *name) {
            Some(m) if t.as_bytes()[m.value.0] == b'{' => obj = m.value.0,
            _ => return Ok(None),
        }
    }
    let (ms, close) = members(t.as_bytes(), obj)?;
    let Some(pos) = ms.iter().position(|m| m.key == key) else { return Ok(None) };
    let m = &ms[pos];
    let end = m.comma_end.unwrap_or(m.value.1);
    if let Some(next) = ms.get(pos + 1) {
        t.replace_range(m.key_start..next.key_start, "");
    } else if pos > 0 {
        t.replace_range(ms[pos - 1].value.1..end, "");
    } else if t[end..close].trim().is_empty() {
        t.replace_range(obj + 1..close, "");
    } else {
        t.replace_range(m.key_start..end, "");
    }
    Ok(Some(t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn up(t: &str) -> String {
        upsert(t, &["mcpServers"], "mem", &json!({"command": "x"})).unwrap()
    }

    #[test]
    fn inserts_into_empty_and_existing_files() {
        let v: Value = serde_json::from_str(&up("")).unwrap();
        assert_eq!(v, json!({"mcpServers": {"mem": {"command": "x"}}}));
        let t = up("{\n  \"theme\": \"dark\",\n  \"mcpServers\": {\n    \"other\": {}\n  }\n}\n");
        assert_eq!(
            t,
            "{\n  \"theme\": \"dark\",\n  \"mcpServers\": {\n    \"other\": {},\n    \"mem\": {\n      \"command\": \"x\"\n    }\n  }\n}\n"
        );
        assert_eq!(up(&t), t, "upsert is idempotent");
    }

    #[test]
    fn keeps_comments_and_trailing_commas() {
        let t = "{\n  // keep me\n  \"a\": 1, /* and me */\n}";
        let out = up(t);
        assert!(out.contains("// keep me") && out.contains("/* and me */"));
        assert!(out.contains("\"a\": 1,\n  \"mcpServers\": {"), "{out}");
    }

    #[test]
    fn removes_members() {
        let t = up("{\"a\": 1}");
        let back = remove(&t, &["mcpServers"], "mem").unwrap().unwrap();
        assert_eq!(serde_json::from_str::<Value>(&back).unwrap(), json!({"a": 1, "mcpServers": {}}));
        let two = "{\"s\": {\"mem\": 1, \"b\": 2}}";
        assert_eq!(remove(two, &["s"], "mem").unwrap().unwrap(), "{\"s\": {\"b\": 2}}");
        let last = "{\"s\": {\"b\": 2, \"mem\": 1}}";
        assert_eq!(remove(last, &["s"], "mem").unwrap().unwrap(), "{\"s\": {\"b\": 2}}");
        assert!(remove(last, &["x"], "mem").unwrap().is_none());
    }

    #[test]
    fn rejects_non_objects() {
        assert!(upsert("[1]", &[], "k", &json!(1)).is_err());
        assert!(upsert("{\"mcpServers\": 3}", &["mcpServers"], "k", &json!(1)).is_err());
    }
}
