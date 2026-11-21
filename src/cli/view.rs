//! Compact text rendering of tool results for the memory commands.
//! Short and line-oriented so agents spend few tokens reading it.

use crate::timefmt::rfc3339;
use serde_json::Value;
use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum View {
    Remember,
    Recall,
    Graph,
    Facts,
    Events,
    Logged,
    Forgot,
    Json,
}

fn s(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}

fn date(ms: &Value) -> String {
    ms.as_u64().map(|m| rfc3339(m)[..10].to_string()).unwrap_or_default()
}

/// Indent continuation lines so multi-line memories stay readable.
fn indented(text: &str, pad: &str) -> String {
    text.trim_end().lines().collect::<Vec<_>>().join(&format!("\n{pad}"))
}

fn fact_line(f: &Value) -> String {
    format!("{} -[{}]-> {}", s(&f["subject"]), s(&f["predicate"]), s(&f["object"]))
}

fn list<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v[key].as_array().map_or(&[], Vec::as_slice)
}

pub fn render(view: View, v: &Value) -> String {
    let mut o = String::new();
    match view {
        View::Remember => {
            let ids: Vec<&str> = list(v, "chunk_ids").iter().map(s).collect();
            let state = if v["created"].as_u64().unwrap_or(0) > 0 {
                "new"
            } else if v["updated"].as_u64().unwrap_or(0) > 0 {
                "updated"
            } else {
                "unchanged"
            };
            let _ = write!(o, "remembered {} ({state}", ids.join(", "));
            match v["facts_added"].as_u64().unwrap_or(0) {
                0 => {}
                1 => o.push_str(", 1 fact"),
                n => {
                    let _ = write!(o, ", {n} facts");
                }
            }
            o.push(')');
        }
        View::Recall => {
            let hits = list(v, "hits");
            if hits.is_empty() {
                o.push_str("no matching memories");
            }
            for (i, h) in hits.iter().enumerate() {
                let tags: Vec<String> = list(h, "tags").iter().map(|t| format!("#{}", s(t))).collect();
                let _ = write!(
                    o,
                    "{}. [{:.2}] {} · {}",
                    i + 1,
                    h["score"].as_f64().unwrap_or(0.0),
                    s(&h["id"]),
                    date(&h["updated_at"])
                );
                if !tags.is_empty() {
                    let _ = write!(o, " · {}", tags.join(" "));
                }
                let _ = writeln!(o, "\n   {}", indented(s(&h["text"]), "   "));
            }
        }
        View::Graph => {
            if v["found"] == false {
                let _ = write!(o, "nothing known about `{}`", s(&v["subject"]));
                return o;
            }
            for f in list(v, "facts") {
                let _ = writeln!(o, "{}", fact_line(f));
            }
            if v["truncated"] == true {
                o.push_str("… (more facts; raise --limit)\n");
            }
            let mems = list(v, "memories");
            if !mems.is_empty() {
                o.push_str("memories:\n");
            }
            for m in mems {
                let _ =
                    writeln!(o, "  {} · {}  {}", s(&m["id"]), date(&m["updated_at"]), indented(s(&m["text"]), "    "));
            }
        }
        View::Facts => {
            let facts = list(v, "facts");
            if facts.is_empty() {
                o.push_str("no matching facts");
            }
            for f in facts {
                let _ = writeln!(o, "{}", fact_line(f));
            }
        }
        View::Events => {
            let events = list(v, "events");
            if events.is_empty() {
                o.push_str("no events");
            }
            for e in events {
                let t = s(&e["time"]).get(..19).unwrap_or("").replace('T', " ");
                let _ = writeln!(
                    o,
                    "{t}  {:<8} {}",
                    s(&e["kind"]),
                    indented(s(&e["text"]), "                               ")
                );
            }
        }
        View::Logged => {
            let _ = write!(o, "logged {}", s(&v["id"]));
        }
        View::Forgot => {
            let n = v["forgotten"].as_u64().unwrap_or(0);
            let _ = write!(o, "forgot {n} memor{}", if n == 1 { "y" } else { "ies" });
        }
        View::Json => o = serde_json::to_string_pretty(v).unwrap_or_default(),
    }
    o.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_recall_and_graph() {
        let r = json!({"hits": [{"id": "ab", "score": 0.5, "updated_at": 0, "tags": ["pref"], "text": "a\nb"}]});
        assert_eq!(render(View::Recall, &r), "1. [0.50] ab · 1970-01-01 · #pref\n   a\n   b");
        assert_eq!(render(View::Recall, &json!({"hits": []})), "no matching memories");
        let g = json!({"found": true, "facts": [{"subject": "u", "predicate": "likes", "object": "tea"}],
            "memories": [{"id": "ab", "updated_at": 0, "text": "u likes tea"}]});
        assert_eq!(render(View::Graph, &g), "u -[likes]-> tea\nmemories:\n  ab · 1970-01-01  u likes tea");
        assert_eq!(render(View::Graph, &json!({"found": false, "subject": "x"})), "nothing known about `x`");
    }

    #[test]
    fn renders_writes() {
        let m = json!({"chunk_ids": ["ab"], "created": 1, "facts_added": 2});
        assert_eq!(render(View::Remember, &m), "remembered ab (new, 2 facts)");
        assert_eq!(render(View::Forgot, &json!({"forgotten": 1})), "forgot 1 memory");
        assert_eq!(render(View::Logged, &json!({"id": "cd"})), "logged cd");
    }
}
