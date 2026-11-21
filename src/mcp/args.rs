//! Lenient, well-reported extraction of tool arguments from JSON.

use crate::error::{Error, Result};
use crate::hash::hex_to_id;
use crate::store::FactInput;
use serde_json::Value;

fn bad(msg: String) -> Error {
    Error::InvalidInput(msg)
}

pub fn opt_str<'a>(a: &'a Value, key: &str) -> Result<Option<&'a str>> {
    match a.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(bad(format!("`{key}` must be a string"))),
    }
}

pub fn req_str<'a>(a: &'a Value, key: &str) -> Result<&'a str> {
    opt_str(a, key)?.ok_or_else(|| bad(format!("missing required argument `{key}`")))
}

pub fn opt_u64(a: &Value, key: &str) -> Result<Option<u64>> {
    match a.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .or_else(|| v.as_f64().filter(|f| *f >= 0.0 && f.is_finite()).map(|f| f as u64))
            .map(Some)
            .ok_or_else(|| bad(format!("`{key}` must be a non-negative number"))),
    }
}

pub fn opt_f64(a: &Value, key: &str) -> Result<Option<f64>> {
    match a.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_f64().map(Some).ok_or_else(|| bad(format!("`{key}` must be a number"))),
    }
}

pub fn opt_bool(a: &Value, key: &str) -> Result<Option<bool>> {
    match a.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(bad(format!("`{key}` must be a boolean"))),
    }
}

pub fn opt_usize(a: &Value, key: &str, default: usize) -> Result<usize> {
    Ok(opt_u64(a, key)?.map_or(default, |v| v.min(usize::MAX as u64) as usize))
}

/// Embedding vector (array of finite numbers).
pub fn opt_vector(a: &Value, key: &str) -> Result<Option<Vec<f32>>> {
    match a.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(xs)) => xs
            .iter()
            .enumerate()
            .map(|(i, x)| x.as_f64().map(|f| f as f32).ok_or_else(|| bad(format!("`{key}[{i}]` is not a number"))))
            .collect::<Result<Vec<f32>>>()
            .map(Some),
        Some(_) => Err(bad(format!("`{key}` must be an array of numbers"))),
    }
}

pub fn opt_str_list(a: &Value, key: &str) -> Result<Option<Vec<String>>> {
    match a.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(vec![s.clone()])),
        Some(Value::Array(xs)) => xs
            .iter()
            .map(|x| x.as_str().map(str::to_string).ok_or_else(|| bad(format!("`{key}` must contain strings"))))
            .collect::<Result<Vec<_>>>()
            .map(Some),
        Some(_) => Err(bad(format!("`{key}` must be an array of strings"))),
    }
}

/// Memory ids as 16-digit hex strings (a single string is accepted too).
pub fn id_list(a: &Value, key: &str) -> Result<Vec<u64>> {
    opt_str_list(a, key)?
        .unwrap_or_default()
        .iter()
        .map(|s| hex_to_id(s).ok_or_else(|| bad(format!("bad id `{s}` in `{key}`"))))
        .collect()
}

/// Facts as objects `{subject, predicate, object}` or triples `[s, p, o]`.
pub fn facts(a: &Value, key: &str) -> Result<Vec<FactInput>> {
    let Some(v) = a.get(key).filter(|v| !v.is_null()) else {
        return Ok(Vec::new());
    };
    let Value::Array(items) = v else {
        return Err(bad(format!("`{key}` must be an array")));
    };
    items
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let parts: Option<[&str; 3]> = match f {
                Value::Array(t) if t.len() == 3 => {
                    Some([t[0].as_str(), t[1].as_str(), t[2].as_str()]).and_then(|[a, b, c]| Some([a?, b?, c?]))
                }
                Value::Object(_) => {
                    (|| Some([f.get("subject")?.as_str()?, f.get("predicate")?.as_str()?, f.get("object")?.as_str()?]))(
                    )
                }
                _ => None,
            };
            let [s, p, o] = parts.ok_or_else(|| {
                bad(format!("`{key}[{i}]` must be {{subject, predicate, object}} or [s, p, o] strings"))
            })?;
            Ok(FactInput { subject: s.into(), predicate: p.into(), object: o.into() })
        })
        .collect()
}

pub fn opt_id(a: &Value, key: &str) -> Result<Option<u64>> {
    match opt_str(a, key)? {
        None => Ok(None),
        Some(s) => hex_to_id(s).map(Some).ok_or_else(|| bad(format!("`{key}` must be a 16-digit hex id"))),
    }
}

/// Duration in milliseconds from seconds (number) or a string such as
/// `"90s"`, `"15m"`, `"12h"`, `"7d"`, `"2w"`.
pub fn parse_duration_ms(v: &Value) -> Result<u64> {
    let err = || bad("durations are seconds or strings like \"30m\", \"12h\", \"7d\"".into());
    let secs = match v {
        Value::Number(n) => n.as_f64().ok_or_else(err)?,
        Value::String(s) => {
            let s = s.trim();
            let split = s.find(|c: char| c.is_alphabetic()).unwrap_or(s.len());
            let (num, unit) = s.split_at(split);
            let n: f64 = num.trim().parse().map_err(|_| err())?;
            let mult = match unit.trim() {
                "" | "s" | "sec" | "secs" => 1.0,
                "m" | "min" | "mins" => 60.0,
                "h" | "hr" | "hrs" => 3600.0,
                "d" | "day" | "days" => 86_400.0,
                "w" | "week" | "weeks" => 604_800.0,
                _ => return Err(err()),
            };
            n * mult
        }
        _ => return Err(err()),
    };
    if !secs.is_finite() || secs < 0.0 {
        return Err(err());
    }
    Ok((secs * 1000.0).min(u64::MAX as f64) as u64)
}

pub fn opt_duration_ms(a: &Value, key: &str) -> Result<Option<u64>> {
    match a.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => parse_duration_ms(v).map(Some),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_durations() {
        assert_eq!(parse_duration_ms(&json!(2)).unwrap(), 2000);
        assert_eq!(parse_duration_ms(&json!("7d")).unwrap(), 7 * 86_400_000);
        assert_eq!(parse_duration_ms(&json!("1.5h")).unwrap(), 5_400_000);
        assert!(parse_duration_ms(&json!("-1d")).is_err());
        assert!(parse_duration_ms(&json!("3 parsecs")).is_err());
        assert!(parse_duration_ms(&json!(true)).is_err());
    }

    #[test]
    fn parses_facts_and_lists() {
        let a = json!({"f": [["a","b","c"], {"subject":"x","predicate":"y","object":"z"}], "t": "solo"});
        let f = facts(&a, "f").unwrap();
        assert_eq!(f[1].object, "z");
        assert!(facts(&json!({"f": [["a", 1, "c"]]}), "f").is_err());
        assert_eq!(opt_str_list(&a, "t").unwrap(), Some(vec!["solo".to_string()]));
        assert!(opt_vector(&json!({"v": [1, "x"]}), "v").is_err());
        assert_eq!(opt_id(&json!({"i": "00000000000000ff"}), "i").unwrap(), Some(255));
        assert!(req_str(&json!({}), "text").is_err());
    }
}
