//! JSON-RPC 2.0 framing (newline-delimited messages over stdio).

use serde_json::{Value, json};

pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;

/// A classified incoming message.
#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    Request {
        id: Value,
        method: String,
        params: Value,
    },
    Notification {
        method: String,
        params: Value,
    },
    /// Not a valid request; carries the error response to send.
    Invalid(Value),
    /// A response from the client (we never send requests; ignored).
    Response,
}

pub fn success(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

pub fn error(id: Value, code: i64, message: &str, data: Option<Value>) -> Value {
    let mut err = json!({"code": code, "message": message});
    if let Some(d) = data {
        err["data"] = d;
    }
    json!({"jsonrpc": "2.0", "id": id, "error": err})
}

fn valid_id(id: &Value) -> bool {
    matches!(id, Value::String(_) | Value::Null) || id.as_i64().is_some() || id.as_u64().is_some()
}

/// Classify one decoded JSON value according to JSON-RPC 2.0.
pub fn classify(v: Value) -> Incoming {
    let Value::Object(mut obj) = v else {
        return Incoming::Invalid(error(Value::Null, INVALID_REQUEST, "request must be an object", None));
    };
    let id = obj.remove("id");
    let reply_id = id.clone().filter(valid_id).unwrap_or(Value::Null);
    if obj.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Incoming::Invalid(error(reply_id, INVALID_REQUEST, "jsonrpc must be \"2.0\"", None));
    }
    if obj.contains_key("result") || obj.contains_key("error") {
        return Incoming::Response;
    }
    let Some(Value::String(method)) = obj.remove("method") else {
        return Incoming::Invalid(error(reply_id, INVALID_REQUEST, "method must be a string", None));
    };
    let params = obj.remove("params").unwrap_or(Value::Null);
    if !matches!(params, Value::Object(_) | Value::Array(_) | Value::Null) {
        return Incoming::Invalid(error(reply_id, INVALID_REQUEST, "params must be structured", None));
    }
    match id {
        None => Incoming::Notification { method, params },
        Some(id) if valid_id(&id) => Incoming::Request { id, method, params },
        Some(_) => Incoming::Invalid(error(Value::Null, INVALID_REQUEST, "invalid id", None)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_messages() {
        let r = classify(json!({"jsonrpc":"2.0","id":1,"method":"ping"}));
        assert!(matches!(r, Incoming::Request { ref method, .. } if method == "ping"));
        let n = classify(json!({"jsonrpc":"2.0","method":"notifications/initialized"}));
        assert!(matches!(n, Incoming::Notification { .. }));
        assert!(matches!(classify(json!([1])), Incoming::Invalid(_)));
        assert!(matches!(classify(json!({"jsonrpc":"1.0","id":1,"method":"x"})), Incoming::Invalid(_)));
        assert!(matches!(classify(json!({"jsonrpc":"2.0","id":{},"method":"x"})), Incoming::Invalid(_)));
        assert!(matches!(classify(json!({"jsonrpc":"2.0","id":1,"method":"x","params":3})), Incoming::Invalid(_)));
        assert_eq!(classify(json!({"jsonrpc":"2.0","id":1,"result":{}})), Incoming::Response);
    }

    #[test]
    fn builds_responses() {
        assert_eq!(success(json!(1), json!({}))["result"], json!({}));
        let e = error(json!("a"), METHOD_NOT_FOUND, "nope", Some(json!({"x": 1})));
        assert_eq!(e["error"]["code"], -32601);
        assert_eq!(e["error"]["data"]["x"], 1);
    }
}
