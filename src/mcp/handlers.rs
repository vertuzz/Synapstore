//! `tools/call` dispatch: argument parsing → store call → JSON result.

use super::args::*;
use crate::decay::DAY_MS;
use crate::error::{Error, Result};
use crate::graph::Direction;
use crate::hash::chunk_id;
use crate::store::explain::ExplainQuery;
use crate::store::*;
use serde::Serialize;
use serde_json::{Value, json};

/// Outcome of resolving a tool name.
pub enum Call {
    Done(Result<Value>),
    UnknownTool,
}

fn to_value<T: Serialize>(v: T) -> Result<Value> {
    serde_json::to_value(v).map_err(|e| Error::Serialization(e.to_string()))
}

/// Execute tool `name` with `args` (an object, or null for no arguments).
pub fn call(store: &SynapStore, name: &str, args: &Value) -> Call {
    let empty = json!({});
    let a = if args.is_null() { &empty } else { args };
    if !a.is_object() {
        return Call::Done(Err(Error::InvalidInput("arguments must be an object".into())));
    }
    Call::Done(match name {
        "memorize" => memorize(store, a),
        "recall" => recall(store, a),
        "graph_query" => graph_query(store, a),
        "graph_match" => (|| {
            let facts = store.graph_match(
                opt_str(a, "subject")?,
                opt_str(a, "predicate")?,
                opt_str(a, "object")?,
                opt_usize(a, "limit", 100)?,
            )?;
            Ok(json!({"count": facts.len(), "facts": to_value(facts)?}))
        })(),
        "add_facts" => (|| {
            let added = store.add_facts(facts(a, "facts")?, opt_id(a, "source_id")?, opt_u64(a, "timestamp_ms")?)?;
            Ok(json!({"facts_added": added}))
        })(),
        "retract_fact" => (|| {
            let f = FactInput {
                subject: req_str(a, "subject")?.into(),
                predicate: req_str(a, "predicate")?.into(),
                object: req_str(a, "object")?.into(),
            };
            Ok(json!({"retracted": store.retract_fact(f)?}))
        })(),
        "log_event" => (|| {
            let e = EventInput {
                kind: opt_str(a, "kind")?.unwrap_or("event").into(),
                text: req_str(a, "text")?.into(),
                timestamp_ms: opt_u64(a, "timestamp_ms")?,
                chunk_id: opt_id(a, "chunk_id")?,
            };
            let ids = store.log_events(&[e])?;
            Ok(json!({"id": crate::hash::id_to_hex(ids[0])}))
        })(),
        "recent_events" => (|| {
            let q = EventQuery {
                limit: opt_usize(a, "limit", 20)?,
                kind: opt_str(a, "kind")?.map(str::to_string),
                since_ms: opt_u64(a, "since_ms")?,
                until_ms: opt_u64(a, "until_ms")?,
            };
            let events = store.recent_events(&q)?;
            Ok(json!({"count": events.len(), "events": to_value(events)?}))
        })(),
        "forget" => (|| {
            let mut ids = Vec::new();
            for s in opt_str_list(a, "ids")?.unwrap_or_default() {
                ids.push(crate::hash::hex_to_id(&s).ok_or_else(|| Error::InvalidInput(format!("bad id `{s}`")))?);
            }
            if let Some(t) = opt_str(a, "text")? {
                ids.push(chunk_id(t.trim()));
            }
            if ids.is_empty() {
                return Err(Error::InvalidInput("provide `ids` or `text`".into()));
            }
            Ok(json!({"forgotten": store.forget(ids)?}))
        })(),
        "explain" => explain(store, a),
        "repair_index" => store.repair_index().and_then(to_value),
        "stats" => opt_bool(a, "detailed").and_then(|d| store.stats(d.unwrap_or(false))),
        _ => return Call::UnknownTool,
    })
}

fn memorize(store: &SynapStore, a: &Value) -> Result<Value> {
    let input = MemorizeInput {
        text: req_str(a, "text")?.into(),
        vector: opt_vector(a, "vector")?,
        entities: opt_str_list(a, "entities")?,
        facts: facts(a, "facts")?,
        tags: opt_str_list(a, "tags")?.unwrap_or_default(),
        source: opt_str(a, "source")?.map(str::to_string),
        kind: opt_str(a, "kind")?.map(str::to_string),
        timestamp_ms: opt_u64(a, "timestamp_ms")?,
    };
    to_value(store.memorize(input)?)
}

fn recall(store: &SynapStore, a: &Value) -> Result<Value> {
    let mut decay = store.config().decay;
    if let Some(days) = opt_f64(a, "half_life_days")? {
        if !(days.is_finite() && days > 0.0) {
            return Err(Error::InvalidInput("`half_life_days` must be > 0".into()));
        }
        decay.half_life_ms = (days * DAY_MS as f64) as u64;
    }
    if let Some(enabled) = opt_bool(a, "decay")? {
        decay.enabled = enabled;
    }
    let q = RecallQuery {
        vector: opt_vector(a, "vector")?,
        text: opt_str(a, "query")?.map(str::to_string),
        top_k: opt_usize(a, "top_k", 5)?,
        max_age_ms: opt_duration_ms(a, "max_age")?,
        now_ms: opt_u64(a, "now_ms")?,
        ef: opt_u64(a, "ef")?.map(|v| v as usize),
        decay: Some(decay),
        tags: opt_str_list(a, "tags")?.unwrap_or_default(),
    };
    to_value(store.recall(&q)?)
}

fn graph_query(store: &SynapStore, a: &Value) -> Result<Value> {
    let mut q = GraphQuery::new(req_str(a, "subject")?);
    q.depth = opt_usize(a, "depth", 2)?;
    if let Some(d) = opt_str(a, "direction")? {
        q.direction =
            Direction::parse(d).ok_or_else(|| Error::InvalidInput("`direction` must be out, in or both".into()))?;
    }
    q.predicate = opt_str(a, "predicate")?.map(str::to_string);
    q.limit = opt_usize(a, "limit", 200)?;
    q.memories = opt_usize(a, "memories", 5)?;
    to_value(store.graph_query(&q)?)
}

fn explain(store: &SynapStore, a: &Value) -> Result<Value> {
    let target = match (opt_id(a, "target_id")?, opt_str(a, "target_text")?) {
        (Some(id), _) => id,
        (None, Some(t)) => chunk_id(t.trim()),
        (None, None) => {
            return Err(Error::InvalidInput("provide `target_id` or `target_text`".into()));
        }
    };
    let q = ExplainQuery {
        target,
        vector: opt_vector(a, "vector")?,
        text: opt_str(a, "query")?.map(str::to_string),
        top_k: opt_usize(a, "top_k", 5)?,
        ef: opt_u64(a, "ef")?.map(|v| v as usize),
        max_age_ms: opt_duration_ms(a, "max_age")?,
        now_ms: opt_u64(a, "now_ms")?,
    };
    to_value(store.explain(&q)?)
}
