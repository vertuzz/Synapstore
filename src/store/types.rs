//! Public request / response types of the store API (serializable so the
//! MCP layer can return them verbatim). Ids are exposed as 16-digit hex.

use crate::decay::DecayParams;
use crate::graph::Direction;
use serde::{Deserialize, Serialize, Serializer};

/// serde helpers rendering u64 ids as hex strings.
pub mod hex {
    use super::Serializer;
    use crate::hash::id_to_hex;

    pub fn one<S: Serializer>(id: &u64, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&id_to_hex(*id))
    }
    pub fn opt<S: Serializer>(id: &Option<u64>, s: S) -> Result<S::Ok, S::Error> {
        match id {
            Some(v) => s.serialize_str(&id_to_hex(*v)),
            None => s.serialize_none(),
        }
    }
    pub fn many<S: Serializer>(ids: &[u64], s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(ids.iter().map(|i| id_to_hex(*i)))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactInput {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

#[derive(Debug, Clone, Default)]
pub struct MemorizeInput {
    pub text: String,
    /// Caller embedding. `None` → built-in lexical embedder (+ chunking).
    pub vector: Option<Vec<f32>>,
    /// Explicit entity names. `None` → heuristic extraction.
    pub entities: Option<Vec<String>>,
    pub facts: Vec<FactInput>,
    pub tags: Vec<String>,
    pub source: Option<String>,
    /// Episode kind recorded in the episodic log (default `"memory"`).
    pub kind: Option<String>,
    pub timestamp_ms: Option<u64>,
    /// Memories this one replaces: forgotten (with the facts they asserted)
    /// in the same transaction.
    pub supersedes: Vec<u64>,
}

impl MemorizeInput {
    pub fn text(text: impl Into<String>) -> Self {
        MemorizeInput { text: text.into(), ..Default::default() }
    }
    pub fn with_vector(mut self, v: Vec<f32>) -> Self {
        self.vector = Some(v);
        self
    }
    pub fn at(mut self, ts_ms: u64) -> Self {
        self.timestamp_ms = Some(ts_ms);
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct MemorizeOutcome {
    #[serde(serialize_with = "hex::many")]
    pub chunk_ids: Vec<u64>,
    pub created: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub entities: Vec<String>,
    pub facts_added: usize,
    pub embedding: String,
    pub dim: usize,
    /// Memories removed through `supersedes`.
    #[serde(skip_serializing_if = "is_zero")]
    pub superseded: usize,
    /// Facts retracted along with the superseded memories.
    #[serde(skip_serializing_if = "is_zero")]
    pub facts_retracted: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

/// Result of `forget`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ForgetOutcome {
    /// Memories that existed and were deleted.
    pub forgotten: usize,
    /// Facts whose provenance was one of those memories.
    pub facts_retracted: usize,
}

#[derive(Debug, Clone, Default)]
pub struct RecallQuery {
    pub vector: Option<Vec<f32>>,
    pub text: Option<String>,
    pub top_k: usize,
    /// Drop memories older than this (milliseconds since last update).
    pub max_age_ms: Option<u64>,
    /// Override "now" (simulation / tests).
    pub now_ms: Option<u64>,
    pub ef: Option<usize>,
    pub decay: Option<DecayParams>,
    /// Keep only chunks carrying at least one of these tags.
    pub tags: Vec<String>,
}

impl RecallQuery {
    /// Search by text with the store's embedder.
    pub fn text(query: impl Into<String>, top_k: usize) -> Self {
        RecallQuery { text: Some(query.into()), top_k, ..Default::default() }
    }
    /// Search by a caller-provided embedding.
    pub fn vector(v: Vec<f32>, top_k: usize) -> Self {
        RecallQuery { vector: Some(v), top_k, ..Default::default() }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecallHit {
    #[serde(serialize_with = "hex::one")]
    pub id: u64,
    pub text: String,
    pub tags: Vec<String>,
    pub entities: Vec<String>,
    pub source: String,
    pub similarity: f32,
    pub score: f32,
    pub decay_factor: f32,
    pub age_days: f64,
    pub created_at: u64,
    pub updated_at: u64,
    pub hits: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct RecallOutcome {
    pub hits: Vec<RecallHit>,
    /// HNSW candidates re-ranked with time-decay.
    pub candidates: usize,
    /// Distance evaluations performed by the index.
    pub visited: u64,
    pub ef: usize,
    pub embedding: String,
    /// Set when the answer is degraded (e.g. the embedding endpoint is down).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphQuery {
    pub subject: String,
    pub depth: usize,
    pub direction: Direction,
    pub predicate: Option<String>,
    pub limit: usize,
    /// How many memories mentioning the subject to attach.
    pub memories: usize,
}

impl GraphQuery {
    pub fn new(subject: impl Into<String>) -> Self {
        GraphQuery {
            subject: subject.into(),
            depth: 2,
            direction: Direction::Both,
            predicate: None,
            limit: 200,
            memories: 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FactView {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub depth: u8,
    pub updated_at: u64,
    #[serde(serialize_with = "hex::opt", skip_serializing_if = "Option::is_none")]
    pub source_chunk: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EntityView {
    pub name: String,
    pub depth: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemoryRef {
    #[serde(serialize_with = "hex::one")]
    pub id: u64,
    pub text: String,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GraphOutcome {
    pub found: bool,
    pub subject: String,
    pub facts: Vec<FactView>,
    pub entities: Vec<EntityView>,
    pub memories: Vec<MemoryRef>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Default)]
pub struct EventInput {
    pub kind: String,
    pub text: String,
    pub timestamp_ms: Option<u64>,
    pub chunk_id: Option<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct EventQuery {
    pub limit: usize,
    pub kind: Option<String>,
    pub since_ms: Option<u64>,
    pub until_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EventView {
    #[serde(serialize_with = "hex::one")]
    pub id: u64,
    pub ts: u64,
    pub time: String,
    pub kind: String,
    pub text: String,
    #[serde(serialize_with = "hex::opt", skip_serializing_if = "Option::is_none")]
    pub chunk_id: Option<u64>,
}
