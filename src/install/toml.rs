//! Line-based edit of one `[mcp_servers.<name>]` table in Codex's
//! `config.toml`; everything else in the file is left byte-for-byte.

/// A TOML basic string.
pub fn string(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if c.is_control() => o.push_str(&format!("\\u{:04X}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn is_header(line: &str) -> bool {
    line.trim_start().starts_with('[')
}

fn owns(line: &str, name: &str) -> bool {
    let h = line.trim();
    let h = h.split('#').next().unwrap_or("").trim();
    [format!("mcp_servers.{name}"), format!("mcp_servers.\"{name}\"")]
        .iter()
        .any(|t| h == format!("[{t}]") || h.starts_with(&format!("[{t}.")))
}

/// Byte range of the table (and its sub-tables), if present.
fn span(text: &str, name: &str) -> Option<(usize, usize)> {
    let mut start = None;
    let mut pos = 0;
    for line in text.split_inclusive('\n') {
        if is_header(line) {
            match (start, owns(line, name)) {
                (None, true) => start = Some(pos),
                (Some(s), false) => return Some((s, pos)),
                _ => {}
            }
        }
        pos += line.len();
    }
    start.map(|s| (s, text.len()))
}

/// Insert or replace the table `[mcp_servers.<name>]` with `body` lines.
pub fn upsert(text: &str, name: &str, body: &str) -> String {
    let table = format!("[mcp_servers.{name}]\n{}\n", body.trim_end());
    match span(text, name) {
        Some((s, e)) => {
            let gap = if e < text.len() { "\n" } else { "" };
            format!("{}{table}{gap}{}", &text[..s], &text[e..])
        }
        None if text.trim().is_empty() => table,
        None => format!("{}\n\n{table}", text.trim_end()),
    }
}

/// Remove the table; `None` when it is not there.
pub fn remove(text: &str, name: &str) -> Option<String> {
    let (s, e) = span(text, name)?;
    let head = text[..s].trim_end();
    let tail = text[e..].trim_start();
    Some(match (head.is_empty(), tail.is_empty()) {
        (true, _) => tail.to_string(),
        (false, true) => format!("{head}\n"),
        (false, false) => format!("{head}\n\n{tail}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upserts_and_removes_one_table() {
        let orig = "model = \"o3\"\n\n[mcp_servers.other]\ncommand = \"x\"\n";
        let body = "command = \"/bin/synapstore\"\nargs = [\"mcp\"]";
        let t = upsert(orig, "synapstore", body);
        assert_eq!(t, format!("{}\n\n[mcp_servers.synapstore]\n{body}\n", orig.trim_end()));
        let again = upsert(&t, "synapstore", "command = \"y\"");
        assert!(again.contains("[mcp_servers.synapstore]\ncommand = \"y\"\n") && !again.contains("/bin/"));
        assert_eq!(remove(&again, "synapstore").unwrap(), orig);
        assert!(remove(orig, "synapstore").is_none());
    }

    #[test]
    fn keeps_following_tables_and_subtables_move_together() {
        let t = "[mcp_servers.synapstore]\nurl = \"a\"\n[mcp_servers.synapstore.env]\nK = \"v\"\n[profile]\nx = 1\n";
        let out = upsert(t, "synapstore", "url = \"b\"");
        assert_eq!(out, "[mcp_servers.synapstore]\nurl = \"b\"\n\n[profile]\nx = 1\n");
        assert_eq!(remove(t, "synapstore").unwrap(), "[profile]\nx = 1\n");
        assert!(upsert("", "s", "a = 1").starts_with("[mcp_servers.s]\na = 1\n"));
    }

    #[test]
    fn escapes_strings() {
        assert_eq!(string(r#"C:\a "b""#), r#""C:\\a \"b\"""#);
    }
}
